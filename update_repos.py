#!/usr/local/bin/python3

"""
Update git repositories from remotes.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import logging.config
import os
import random
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import chain
from typing import Any, ParamSpec, Self, TypeAlias, TypeVar

import git

__all__ = [
    "MyExecutor",
    "Repo",
    "no_traceback_log_filter",
    "depthwalk",
    "parse_arguments",
    "main",
]

T = TypeVar("T")
C = TypeVar("C", bound=Callable)
P = ParamSpec("P")

# akin to `Optional`, but with Exceptions instead of `None`
ExceptionOr: TypeAlias = Exception | T

# the triplets that `os.walk` & friends yield
WalkEntry: TypeAlias = tuple[str, list[str], list[str]]

# things that resolve to a commit
CommitRef: TypeAlias = git.Commit | git.Tag | git.Head

FETCHINFO_FLAGS = {
    git.FetchInfo.ERROR: "ERROR",
    git.FetchInfo.FAST_FORWARD: "FAST_FORWARD",
    git.FetchInfo.FORCED_UPDATE: "FORCED_UPDATE",
    git.FetchInfo.HEAD_UPTODATE: "HEAD_UPTODATE",
    git.FetchInfo.NEW_HEAD: "NEW_HEAD",
    git.FetchInfo.NEW_TAG: "NEW_TAG",
    git.FetchInfo.REJECTED: "REJECTED",
    git.FetchInfo.TAG_UPDATE: "TAG_UPDATE",
}

REPO_DIR: ContextVar[str | None] = ContextVar("REPO_DIR", default=None)

logger = logging.getLogger("update-repos")


class ContextAddedLogFilter(logging.Filter):
    """A logging filter that populates a LogRecord attribute from a ContextVar."""

    def __init__(self, *args, var: ContextVar, attribute: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.attribute = attribute

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, self.attribute, self.var.get())
        return super().filter(record)


def no_traceback_log_filter(record: logging.LogRecord) -> bool:
    """log filter function to unconditionally remove exception info"""
    record.exc_info = None
    return True


class MyExecutor(ThreadPoolExecutor):
    """A subclass of `ThreadPoolExecutor` that keeps track of all futures it
    creates, and provides access to them via a generator method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._futures = []

    def submit(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        ftr = super().submit(fn, *args, **kwargs)
        self._futures.append(ftr)
        return ftr

    @property
    def futures(self) -> Generator[Future]:
        yield from self._futures


class Repo(git.Repo):
    """Just like git.Repo, but with more"""

    def __init__(self, path: str, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.orig_path = path
        self._sibling_worktrees: list[Self] | None = None

    @classmethod
    def find_recursive(
        cls,
        starting_path: str,
        *,
        max_depth: int | None = None,
        exclude: Iterable[str] | None = None,
        include_ancestors: bool = True,
        force_recurse: bool = False,
        absolute_paths: bool = False,
    ) -> Generator[Self]:
        """
        Enumerate Git Repositories.

        Search first in and beneath the given `path` (optionally limited to a
        maximum depth), and then optionally also upwards through the path's
        ancestors. Skip searching paths matching optional wildcard exclusion
        patterns. Do not recurse into found repositories unless `force_recurse`
        is True. Build enumerated repositories from relative paths unless
        `absolute_paths` is True.
        """

        starting_path = os.path.abspath(starting_path) if absolute_paths else os.path.normpath(starting_path)
        logger.debug("%s: finding git repositories: (maxdepth=%s)", starting_path, max_depth)

        walk_generator = depthwalk(starting_path, maxdepth=max_depth)
        if include_ancestors:
            walk_generator = chain(walk_generator, backwalk(starting_path))

        exclusion_patterns = cls._build_exclusion_patterns(exclude) if exclude else []

        for path, dirs, files in walk_generator:
            if any(fnmatch.fnmatch(path, pat) for pat in exclusion_patterns):
                logger.debug("Excluding path: %s", path)
                # clearing `dirs` prevents `os.walk` (and hence `depthwalk`)
                # from traversing any deeper, but doesn't affect `backwalk`
                dirs.clear()
                continue

            if ".git" in dirs or ".git" in files:
                try:
                    yield cls(path)
                except git.GitError as exc:
                    logger.warning("Skipping path: %r", exc)
                if not force_recurse:
                    dirs.clear()
                    continue

    @staticmethod
    def _build_exclusion_patterns(patterns: Iterable[str]) -> list[str]:
        """Build normalized exclusion patterns"""
        # possibilities for an exclusion pattern:
        # a single directory component (".git")
        # a partial path (".git/config")
        # a full relative path ("./workspace/mmicro")
        # ... an absolute path? ("/opt/dev/core.git")

        # i think this boils down to being either a path or a path fragment
        # i think we can detect paths by checking if a pattern starts with any
        # of "./", "../", or "/", ... or if it exactly matches "." or ".."

        exclusions: set[str] = set()
        for pattern in patterns:
            if pattern in (".", ".."):
                pattern += "/"
            if any(pattern.startswith(prefix) for prefix in ("./", "../", "/")):
                exclusions.add(os.path.normpath(pattern))
            else:
                exclusions.add("*/" + pattern)

        return list(exclusions)

    def is_ancestor(self, ancestor_rev: CommitRef, rev: CommitRef) -> bool:
        return super().is_ancestor(ancestor_rev, rev)  # type: ignore

    def update(self, *, lprune: bool = False) -> bool:
        with context(REPO_DIR, self.orig_path):
            logger.info("Updating repository")
            failures = False
            if self.is_main_worktree:
                failures |= self.update_remotes()
                failures |= self.update_branches(lprune=lprune)
            else:
                failures |= self.update_checked_out_branch()
            return failures

    def update_remotes(self) -> bool:
        with context(REPO_DIR, self.orig_path):
            failures = False
            for remote in self.remotes:
                if remote.config_reader.get("skipFetchAll", fallback=None):
                    continue
                logger.debug("fetching: %s", remote)
                start = time.monotonic()
                while True:
                    fetchinfolist = []
                    try:
                        fetchinfolist = remote.fetch()
                        break
                    except git.GitCommandError as exc:
                        logger.exception("%s", exc)
                        if time.monotonic() - start > 120:
                            logger.error("caught the thing too many times; giving up")
                            failures = True
                            break
                        else:
                            logger.warning("caught the thing; sleeping...")
                            time.sleep(5)
                    except Exception:
                        logger.exception("Failed to fetch: %s", remote)
                        failures = True
                        break

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "fetchinfolist: %s: %s",
                        remote,
                        self.fetchinfolist_to_str(fetchinfolist),
                    )
            return failures

    def update_branches(self, *, lprune: bool = False) -> bool:
        """update all local branches from their respective tracking branches (if set)."""
        with context(REPO_DIR, self.orig_path):
            failures = False
            for branch in self.branches:
                try:
                    failures |= self.update_branch(branch, lprune=lprune)
                except Exception as exc:
                    logger.exception("Failed to update branch: %s (%s)", branch, exc)
                    failures = True

            return failures

    def update_branch(self, branch: git.Head, *, lprune: bool = False) -> bool:
        """fast-forward the given local branch from its tracking branch, if set."""
        with context(REPO_DIR, self.orig_path):
            failure = False
            checked_out = self.branch_is_checked_out_anywhere(branch)
            tracking_branch = branch.tracking_branch()
            if tracking_branch is None:
                logger.info("no tracking branch set: %s", branch)
            elif not tracking_branch.is_valid():
                if (not checked_out) and lprune:
                    logger.info("Pruning local branch: %s", branch)
                    self.delete_head(branch, force=True)
                else:
                    failure = True
                    logger.warning(
                        "Failed to update branch; tracking branch gone upstream: %s",
                        branch,
                    )
            elif self.is_ancestor(tracking_branch, branch):
                logger.debug("no changes for branch: %s", branch)
            elif not self.is_ancestor(branch, tracking_branch):
                failure = True
                logger.warning(
                    "Failed to update branch; not fast-forward: %s",
                    branch,
                )
            elif checked_out:
                # branch is checked out *somewhere*, but not necessarily here...
                if self.head.is_detached or self.head.ref != branch:
                    # its checked out somewhere else!
                    logger.info("skipping branch checked out elsewhere: %s", branch)
                elif self.is_dirty():
                    failure = True
                    logger.warning("Refusing to update dirty checked out branch: %s", branch)
                else:
                    logger.info("fast-forwarding checked out branch: %s", branch)
                    self.head.reset(tracking_branch, index=True, working_tree=True)
            else:
                branch.commit = tracking_branch.commit
                logger.info("Updated branch: %s", branch)
            return failure

    def update_checked_out_branch(self) -> bool:
        """update the currently checked out branch of this repository"""
        with context(REPO_DIR, self.orig_path):
            failure = False
            if self.head.is_detached:
                logger.info("repo is in detached state")
            elif not isinstance(self.head.ref, git.Head):
                logger.warning("%s: HEAD not pointed at a Head?: %s", self.head.ref)
            else:
                failure = self.update_branch(self.head.ref)
            return failure

    @property
    def worktrees(self) -> list[Self]:
        """
        given a git Repo, return a list of Repos representing all worktrees related
        to that Repo (including the passed Repo)
        """
        if self._sibling_worktrees:
            return self._sibling_worktrees

        porcelain: str = self.git.worktree("list", "--porcelain")

        # key-value pairs related to a single worktree
        parts = {}
        # a collection of parts
        parts_list = [parts]
        for line in porcelain.splitlines():
            if not line:
                # start of a new group of parts
                parts = {}
                parts_list.append(parts)
                continue
            lineparts = line.split(" ", maxsplit=1)
            if len(lineparts) == 2:
                key, value = lineparts
            else:
                key, value = line, True
            parts[key] = value

        worktrees = []
        RepoType = type(self)
        for parts in parts_list:
            worktree_dir = parts.get("worktree")
            if not worktree_dir:
                logger.warning("Malformed worktree: %s", parts)
            elif worktree_dir == self.working_dir:
                worktrees.append(self)
            else:
                worktree = RepoType(worktree_dir)
                worktrees.append(worktree)

        if self not in worktrees:
            raise RuntimeError("Own worktree not found in worktrees list", self, porcelain)

        for worktree in worktrees:
            worktree.worktrees = worktrees

        return self.worktrees

    @worktrees.setter
    def worktrees(self, worktrees: list[Self]):
        if self not in worktrees:
            raise ValueError("Repo must be among its own sibling worktrees", self, worktrees)
        self._sibling_worktrees = list(worktrees)

    @property
    def is_main_worktree(self) -> bool:
        return self.git_dir == self.common_dir

    def branch_is_checked_out_anywhere(self, branch: git.Head) -> bool:
        """
        find out whether this branch is checked out in this worktree or any related
        worktree
        """

        for worktree in self.worktrees:
            # can't do a simple equality check on `branch`, b/c we may be from different Repos
            if not worktree.head.is_detached and worktree.head.ref.path == branch.path:
                # this branch is checked out in this worktree
                return True
        return False

    @classmethod
    def fetchinfo_to_str(cls, fetchinfo: git.FetchInfo) -> str:
        result = str(fetchinfo.ref)
        remote_ref = str(fetchinfo.remote_ref_path or "").strip()
        if remote_ref:
            result = f"({remote_ref}) {result}"
        if fetchinfo.flags & (git.FetchInfo.FORCED_UPDATE | git.FetchInfo.FAST_FORWARD):
            result = f"{result} <- {fetchinfo.old_commit}"
        flag_list = [flag_name for flag, flag_name in FETCHINFO_FLAGS.items() if flag & fetchinfo.flags]
        flags = " | ".join(flag_list)
        result = f"{result} ({flags})"
        return result

    @classmethod
    def fetchinfolist_to_str(
        cls,
        fetchinfolist: list[git.FetchInfo],
        *,
        indent: int = 4,
        show_all: bool = False,
    ) -> str:
        # remove boring items
        if not show_all:
            fetchinfolist = [item for item in fetchinfolist if item.flags ^ git.FetchInfo.HEAD_UPTODATE]
        # short-circuit for boring lists
        if not fetchinfolist:
            return "[]"
        # translate each
        lines = [cls.fetchinfo_to_str(item) for item in fetchinfolist]
        separator = "\n"
        if indent:
            separator += " " * indent

        return separator + separator.join(lines)


def depthwalk(
    starting_path: str,
    onerror: Callable[[OSError], Any] | None = None,
    topdown: bool = True,
    followlinks: bool = False,
    maxdepth: int | None = None,
) -> Generator[WalkEntry]:
    """just like `os.walk`, but with new added `maxdepth` parameter!"""
    walk_generator = os.walk(top=starting_path, topdown=topdown, onerror=onerror, followlinks=followlinks)
    if maxdepth is None:
        yield from walk_generator
        return

    depthmap = {starting_path: 0}
    for path, dirs, files in walk_generator:
        yield path, dirs, files
        here = depthmap[path]
        if here >= maxdepth:
            dirs.clear()
            continue
        for dir in dirs:
            child = os.path.join(path, dir)
            depthmap[child] = here + 1


def backwalk(starting_path: str) -> Generator[WalkEntry]:
    """similar to `os.walk`, but goes up instead of down"""
    curr = os.path.abspath(starting_path)
    while True:
        parent = os.path.dirname(curr)
        if not parent or curr == parent:
            break
        curr = parent

        dirs, files = [], []
        for item in os.scandir(curr):
            (dirs if item.is_dir() else files).append(item.name)

        yield curr, dirs, files


@contextmanager
def context(var: ContextVar[T], value: T):
    """Set a context variable within a context block."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        default=["."],
        help="directories to search for repositories (defaults to current directory)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        help="maximum depth to search for repositories",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        help="directories to exclude from consideration; may be repeated; accepts wildcards (remember to quote them!)",
    )
    parser.add_argument(
        "-R",
        "--force-recurse",
        action="store_true",
        help="recurse into repositories, updating submodules & simple nested repositories",
    )
    parser.add_argument(
        "-1",
        "--single-thread",
        action="store_true",
        help="Run in a single thread. useful for background tasks, or debugging.",
    )

    # action_group = parser.add_mutually_exclusive_group()
    # action_group.add_argument(
    #     "-u",
    #     "--update",
    #     action="store_true",
    #     default=True,
    #     help="update repositories (the default)",
    # )
    # action_group.add_argument(
    #     "-s",
    #     "--status",
    #     action="store_true",
    #     help="report the status of located repositories",
    # )

    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="don't update; print the paths of located repositories",
    )

    parser.add_argument(
        "-l",
        "--lprune",
        action="store_true",
        help="If a local branch tracks a remote branch that no longer exists, prune the local branch",
    )

    parser.add_argument(
        "-s",
        "--slow",
        action="store_true",
        help="try to avoid ratelimit. implies `--single-thread`",
    )

    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument(
        "-r",
        "--random-order",
        action="store_true",
        help="operate on repositories in a randomized order",
    )
    order_group.add_argument(
        "-S",
        "--sorted",
        action="store_true",
        help="operate on repositories in alphabetical order",
    )

    parser.add_argument(
        "-a",
        "--absolute",
        action="store_true",
        help="make repository paths absolute before operating on them",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase the verbosity of the script",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="decrease the verbosity of the script",
    )

    traceback_group = parser.add_mutually_exclusive_group()
    traceback_group.add_argument(
        "-t",
        "--traceback",
        action="store_true",
        help="display tracebacks on error (defaults to True when verbosity >= 2)",
    )
    traceback_group.add_argument(
        "-T",
        "--no-traceback",
        action="store_true",
        help="disable tracebacks (even when verbosity >= 2)",
    )

    return parser.parse_args(argv)


def setup_logging(quiet: int, verbose: int, tracebacks: bool | None) -> None:
    volume = verbose - quiet

    # if volume <= -2:
    #     level = logging.FATAL
    # elif volume <= -1:
    #     level = logging.CRITICAL
    # elif volume <= 0:
    #     level = logging.WARNING
    # elif volume <= 1:
    #     level = logging.INFO
    # elif volume <= 2:
    #     level = logging.DEBUG

    level = logging.WARNING - (volume * 10)
    level = min(max(logging.DEBUG, level), logging.FATAL)

    filters = []
    if tracebacks is False:
        filters.append(no_traceback_log_filter)
    elif tracebacks is None and volume <= 1:
        filters.append(no_traceback_log_filter)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(levelname)8s %(asctime)s] %(repo_dir)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "filters": {
                "repo_dir_context": {
                    "()": ContextAddedLogFilter,
                    "var": REPO_DIR,
                    "attribute": "repo_dir",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "standard",
                    "filters": ["repo_dir_context"] + filters,
                    "class": "logging.StreamHandler",
                },
            },
            "loggers": {
                "update-repos": {
                    "handlers": ["console"],
                    "level": level,
                    "propagate": False,
                },
            },
        }
    )


def collect_repos(args: argparse.Namespace) -> Generator[Repo]:
    if args.random_order:
        args.random_order = False
        repos = list(collect_repos(args))
        random.shuffle(repos)
        yield from repos
        return

    seen: set[Repo] = set()
    for directory in args.directories:
        for repo in Repo.find_recursive(
            starting_path=directory,
            max_depth=args.depth,
            exclude=args.exclude,
            include_ancestors=True,
            force_recurse=args.force_recurse,
            absolute_paths=args.absolute,
        ):
            if repo in seen:
                continue
            seen.add(repo)
            yield repo


def _single_threaded_main(args: argparse.Namespace) -> bool:
    """
    Update git repositories from remotes, one at a time.
    Returns a boolean indicating whether errors occurred.
    """

    any_errors = False

    if args.print:
        for repo in collect_repos(args):
            print(repo.orig_path)
        return any_errors

    def process(func: Callable[P, bool], *func_args: P.args, **func_kwargs: P.kwargs):
        if args.slow:
            time.sleep(1)
        try:
            return func(*func_args, **func_kwargs)
        except Exception as exc:
            logger.exception("Caught an exception: %s", exc)
            return True

    linked_worktrees: list[Repo] = []
    for repo in collect_repos(args):
        if not repo.is_main_worktree:
            linked_worktrees.append(repo)
            continue
        any_errors |= process(repo.update, lprune=args.lprune)

    for repo in linked_worktrees:
        any_errors |= process(repo.update, lprune=args.lprune)

    return any_errors


def _multi_threaded_main(args: argparse.Namespace) -> bool:
    """update git repositories from remotes in parallel."""
    any_errors = False
    linked_worktrees = []

    with MyExecutor() as executor:
        for repo in collect_repos(args):
            if repo.is_main_worktree:
                executor.submit(repo.update, lprune=args.lprune)
            else:
                linked_worktrees.append(repo)

        for ftr in as_completed(executor.futures):
            try:
                any_errors |= ftr.result()
            except Exception as exc:
                logger.exception("Caught an exception: %s", exc)
                any_errors = True

        for repo in linked_worktrees:
            try:
                repo.update_checked_out_branch()
            except Exception as exc:
                logger.exception("Caught an exception: %s", exc)
                any_errors = True

    return any_errors


def main(argv: Sequence[str] | None = None) -> None:
    """Update git repositories from remotes."""

    args = parse_arguments(argv)
    setup_logging(args.quiet, args.verbose, args.traceback)

    logger.debug("Args: %r", args)

    os.environ["GIT_ASKPASS"] = "echo"

    try:
        if args.single_thread or args.print or args.slow:
            failures = _single_threaded_main(args)
        else:
            failures = _multi_threaded_main(args)
    except KeyboardInterrupt:
        logger.fatal("Caught interrupt")
        exit(1)
    except Exception as exc:
        logger.fatal("Fatal error: %s", exc, exc_info=True)
        exit(1)

    exit(failures)


if __name__ == "__main__":
    main()
