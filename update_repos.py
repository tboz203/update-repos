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
from typing import Any, Optional, ParamSpec, Self, TypeAlias, TypeVar, Union

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
P = ParamSpec("P")

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

REPO_DIR: ContextVar[str] = ContextVar("REPO_DIR", default="(no repo set)")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(levelname)8s %(asctime)s] %(repo_dir)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "verbose": {
            "format": (
                "[%(levelname)8s %(asctime)s %(threadName)s %(name)s] %(repo_dir)s: %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "repo_dir_context": {
            "()": f"{__name__}.ContextAddedFilter",
            "var": REPO_DIR,
            "attribute": "repo_dir",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "standard",
            "filters": ["repo_dir_context"],
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "update-repos": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

# the triplets that `os.walk` & friends yield
WalkEntry: TypeAlias = tuple[str, list[str], list[str]]

# things that resolve to a commit
CommitRef: TypeAlias = Union[git.Commit, git.Tag, git.Head]

logger = logging.getLogger("update-repos")


class ContextAddedFilter(logging.Filter):
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

    def submit(self, *args, **kwargs) -> Future:
        ftr = super().submit(*args, **kwargs)
        self._futures.append(ftr)
        return ftr

    def futures(self) -> Generator[Future]:
        yield from self._futures


class Repo(git.Repo):
    """Just like git.Repo, but with more"""

    def __init__(
        self,
        pathlike,
        *args,
        **kwargs,
    ):
        super().__init__(pathlike, *args, **kwargs)
        self.orig_path = str(pathlike)
        self._sibling_worktrees: Optional[list[Self]] = None

    @classmethod
    def find_recursive(
        cls,
        root: str,
        maxdepth: Optional[int] = None,
        exclude: Optional[Iterable[str]] = None,
        force_recurse: bool = False,
        absolute_paths: bool = False,
    ) -> Generator[Self]:
        """
        find git repositories above and beneath a root directory, optionally
        limiting max depth. (will not return submodules)
        """

        if absolute_paths:
            root = os.path.abspath(root)
        else:
            root = os.path.normpath(root)

        logger.debug("%s: finding git repositories: (maxdepth=%s)", root, maxdepth)

        if maxdepth is None:
            walk_generator = os.walk(root)
        else:
            walk_generator = depthwalk(root, maxdepth=maxdepth)

        # include ancestors of this directory, from low to high
        walk_generator = chain(walk_generator, backwalk(root))

        if not exclude:
            exclude = []

        # possibilities for an exclusion pattern:
        # a single directory component (".git")
        # a partial path (".git/config")
        # a full relative path ("./workspace/mmicro")
        # ... an absolute path? ("/opt/dev/core.git")

        # i think this boils down to being either a path or a path fragment
        # i think we can detect paths by checking if a pattern starts with any
        # of "./", "../", or "/", ... or if it exactly matches "." or ".."

        new_exclude: list[str] = []
        for pattern in exclude:
            if pattern in [".", ".."]:
                pattern = pattern + "/"
            if any(pattern.startswith(prefix) for prefix in ("./", "../", "/")):
                new_exclude.append(os.path.normpath(pattern))
            else:
                new_exclude.append("*/" + pattern)

        for path, dirs, files in walk_generator:
            if ".git" in dirs or ".git" in files:
                yield cls(path)
                if not force_recurse:
                    # clearing `dirs` prevents `os.walk` from traversing any deeper
                    dirs.clear()

            long_short_map = {os.path.join(path, dir): dir for dir in dirs}

            for pattern in new_exclude:
                for match in fnmatch.filter(long_short_map.keys(), pattern):
                    dirs.remove(long_short_map[match])

    def is_ancestor(self, ancestor_rev: CommitRef, rev: CommitRef) -> bool:
        return super().is_ancestor(ancestor_rev, rev)  # type: ignore

    def update(self, lprune: bool = False) -> bool:
        with context(REPO_DIR, self.orig_path):
            logger.info("Updating repository")
            failures = False
            if self.is_main_worktree:
                failures |= self.update_remotes()
                failures |= self.update_branches(lprune)
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

    def update_branches(self, lprune: bool = False) -> bool:
        """update all local branches from their respective tracking branches (if set)."""
        with context(REPO_DIR, self.orig_path):
            failures = False
            for branch in self.branches:
                try:
                    failures |= self.update_branch(branch, lprune)
                except Exception as exc:
                    logger.exception("Failed to update branch: %s (%s)", branch, exc)
                    failures = True

            return failures

    def update_branch(self, branch: git.Head, lprune: bool = False) -> bool:
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
                    logger.warning(
                        "Refusing to update dirty checked out branch: %s", branch
                    )
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
            if worktree_dir == self.working_dir:
                worktrees.append(self)
                continue
            worktree = RepoType(worktree_dir)
            worktrees.append(worktree)

        if self not in worktrees:
            raise RuntimeError(
                "Own worktree not found in worktrees list", self, worktrees
            )

        for worktree in worktrees:
            worktree.worktrees = worktrees

        return self.worktrees

    @worktrees.setter
    def worktrees(self, worktrees):
        if self not in worktrees:
            raise ValueError(
                "Repo must be among its own sibling worktrees", self, worktrees
            )
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
            if (
                not worktree.head.is_detached
            ) and worktree.head.ref.path == branch.path:
                # this branch is checked out in this worktree
                return True
        return False

    @classmethod
    def fetchinfo_to_str(cls, fetchinfo: git.FetchInfo) -> str:
        out = str(fetchinfo.ref)
        if (remote_ref := fetchinfo.remote_ref_path) and (
            remote_ref := str(remote_ref).strip()
        ):
            out = f"({remote_ref}) {out}"
        if fetchinfo.flags & (git.FetchInfo.FORCED_UPDATE | git.FetchInfo.FAST_FORWARD):
            out = f"{out} <- {fetchinfo.old_commit}"
        flag_list = [
            flag_name
            for flag, flag_name in FETCHINFO_FLAGS.items()
            if flag & fetchinfo.flags
        ]
        flags = " | ".join(flag_list)
        out = f"{out} ({flags})"
        return out

    @classmethod
    def fetchinfolist_to_str(
        cls, fetchinfolist: list[git.FetchInfo], indent=4, show_all=False
    ) -> str:
        # remove boring items
        if not show_all:
            fetchinfolist = [
                item
                for item in fetchinfolist
                if item.flags ^ git.FetchInfo.HEAD_UPTODATE
            ]
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
    top: str,
    topdown: bool = True,
    onerror: Optional[Callable[[OSError], Any]] = None,
    followlinks: bool = False,
    maxdepth: Optional[int] = None,
) -> Generator[WalkEntry]:
    """just like `os.walk`, but with new added `maxdepth` parameter!"""
    if maxdepth is None:
        yield from os.walk(top, topdown, onerror, followlinks)
        return

    depthmap = {top: 0}
    for path, dirs, files in os.walk(top, topdown, onerror, followlinks):
        yield path, dirs, files
        here = depthmap[path]
        if here >= maxdepth:
            dirs.clear()
            continue
        for dir in dirs:
            child = os.path.join(path, dir)
            depthmap[child] = here + 1


def backwalk(top: str) -> Generator[WalkEntry]:
    """just like `os.walk`, but goes up instead of down"""
    curr = os.path.abspath(top)
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
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        default=["."],
        help="directories to search for repositories (defaults to current directory)",
    )
    parser.add_argument(
        "-d", "--depth", type=int, help="maximum depth to search for repositories"
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


def setup_logging(volume: int, tracebacks: Optional[bool]) -> None:
    logging.config.dictConfig(LOGGING_CONFIG)

    if volume <= -2:
        logger.setLevel(logging.FATAL)
    elif volume <= -1:
        logger.setLevel(logging.CRITICAL)
    elif volume <= 0:
        logger.setLevel(logging.WARNING)
    elif volume <= 1:
        logger.setLevel(logging.INFO)
    elif volume <= 2:
        logger.setLevel(logging.DEBUG)

    if tracebacks is False:
        logger.addFilter(no_traceback_log_filter)
    elif tracebacks is None and volume <= 1:
        logger.addFilter(no_traceback_log_filter)


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
            directory, args.depth, args.exclude, args.force_recurse, args.absolute
        ):
            if repo in seen:
                continue
            seen.add(repo)
            yield repo


def _main_singlethreaded(args: argparse.Namespace) -> bool:
    """update git repositories from remotes, one at a time."""
    failures = False

    if args.print:
        for repo in collect_repos(args):
            print(repo.orig_path)
        return failures

    def process(func: Callable[P, bool], *func_args: P.args, **func_kwargs: P.kwargs):
        nonlocal failures
        if args.slow:
            time.sleep(1)
        try:
            failures |= func(*func_args, **func_kwargs)
        except Exception as exc:
            logger.exception("Caught an exception: %s", exc)
            failures = True

    linked_worktrees: list[Repo] = []
    for repo in collect_repos(args):
        if not repo.is_main_worktree:
            linked_worktrees.append(repo)
            continue
        process(repo.update, args.lprune)

    for repo in linked_worktrees:
        process(repo.update, args.lprune)

    return failures


def _main_multithreaded(args: argparse.Namespace) -> bool:
    failures = False

    with MyExecutor() as executor:
        linked_worktrees = []
        for repo in collect_repos(args):
            if repo.is_main_worktree:
                executor.submit(repo.update, args.lprune)
            else:
                linked_worktrees.append(repo)

        for ftr in as_completed(executor.futures()):
            try:
                failures |= ftr.result()
            except Exception as exc:
                logger.exception("Caught an exception: %s", exc)
                failures = True

        for repo in linked_worktrees:
            try:
                repo.update_checked_out_branch()
            except Exception as exc:
                logger.exception("Caught an exception: %s", exc)
                failures = True

    return failures


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Update git repositories from remotes."""

    args = parse_arguments(argv)
    setup_logging(volume=args.verbose - args.quiet, tracebacks=args.traceback)

    os.environ["GIT_ASKPASS"] = "echo"

    try:
        if args.single_thread or args.print or args.slow:
            failures = _main_singlethreaded(args)
        else:
            failures = _main_multithreaded(args)
    except KeyboardInterrupt:
        logger.warning("Caught interrupt")
        exit(1)

    exit(failures)


if __name__ == "__main__":
    main()
