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
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Optional, Self, TypeAlias, Union

import git

__all__ = [
    "MyExecutor",
    "Repo",
    "no_traceback_log_filter",
    "depthwalk",
    "parse_arguments",
    "main",
]

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": ("[%(levelname)8s %(asctime)s] %(message)s"),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "verbose": {
            "format": ("[%(levelname)8s %(asctime)s %(threadName)s %(name)s] %(message)s"),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "standard",
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

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("update-repos")

# the triplets that `os.walk` & friends yield
WalkEntry: TypeAlias = tuple[str, list[str], list[str]]

# things that resolve to a commit
CommitRef: TypeAlias = Union[git.Commit, git.Tag, git.Head]


def no_traceback_log_filter(record: logging.LogRecord) -> bool:
    """log filter function to unconditionally remove exception info"""
    record.exc_info = None
    return True


# removing these for now b/c unsuitable for multithreaded execution

# def add_prefix_log_filter_factory(prefix: str) -> Callable[[logging.LogRecord], bool]:
#     """create a log filter function that prefixes all log messages with a string constant"""
#     def log_filter(record: logging.LogRecord) -> bool:
#         record.message = prefix + record.message
#         # the filter will not drop any records
#         return True
#     return log_filter


# @contextmanager
# def log_prefix(logger: logging.Logger, prefix: str):
#     """context manager to prefix messages sent to the given logger with the given message"""
#     log_filter = add_prefix_log_filter_factory(prefix)
#     logger.addFilter(log_filter)
#     yield
#     logger.removeFilter(log_filter)


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

    def futures(self) -> Iterator[Future]:
        yield from self._futures


def depthwalk(top: str, maxdepth: int = -1, **kwargs) -> Iterator[WalkEntry]:
    """just like `os.walk`, but with new added `maxdepth` parameter! negative value means 'no limit'."""
    depthmap: dict[str, int] = {top: 0}
    for path, dirs, files in os.walk(top, **kwargs):
        yield path, dirs, files
        here = depthmap[path]
        if here >= maxdepth >= 0:
            dirs.clear()
            continue
        for dir in dirs:
            child = os.path.join(path, dir)
            depthmap[child] = here + 1


def backwalk(top: str) -> Iterator[WalkEntry]:
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


class Repo(git.Repo):
    """Just like git.Repo, but with more"""

    def __init__(self, pathlike, *args, **kwargs):
        super().__init__(pathlike, *args, **kwargs)
        # keep track of the path that was passed to us
        self.orig_path = pathlike

    @classmethod
    def find_recursive(
        cls: type[Self],
        root: str,
        maxdepth: Optional[int] = None,
        exclude: Optional[Sequence[str]] = None,
        randomized: bool = False,
        force_recurse: bool = False,
        absolute_paths: bool = False,
    ) -> list[Self]:
        """
        find git repositories above and beneath a root directory, optionally
        limiting max depth. (will not return submodules)
        """

        logger.debug("%s: finding git repositories: (maxdepth=%s)", root, maxdepth)

        if absolute_paths:
            root = os.path.abspath(root)
        else:
            root = os.path.normpath(root)

        if maxdepth is None:
            walk_generator = os.walk(root)
        else:
            walk_generator = depthwalk(root, maxdepth)

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

        repos: list[Self] = []

        for path, dirs, files in walk_generator:
            if ".git" in dirs or ".git" in files:
                repos.append(cls(path))
                if not force_recurse:
                    # clearing `dirs` prevents `os.walk` from traversing any deeper
                    dirs.clear()

            long_short_map = {os.path.join(path, dir): dir for dir in dirs}

            for pattern in new_exclude:
                for match in fnmatch.filter(long_short_map.keys(), pattern):
                    dirs.remove(long_short_map[match])

        if randomized:
            random.shuffle(repos)

        return repos

    def is_ancestor(self, ancestor_rev: CommitRef, rev: CommitRef) -> bool:
        return super().is_ancestor(ancestor_rev, rev)  # type: ignore

    def update_branch(self: Self, branch: git.Head, lprune: bool = False) -> None:
        """if a branch has a remote tracking branch, update it."""

        checked_out = self.branch_is_checked_out_anywhere(branch)
        tracking_branch = branch.tracking_branch()
        if tracking_branch is None:
            logger.debug("%s: no tracking branch set: %s", self.orig_path, branch)
        elif not tracking_branch.is_valid():
            if (not checked_out) and lprune:
                logger.info("%s: Pruning local branch: %s", self.orig_path, branch)
                self.delete_head(branch, force=True)
            else:
                logger.warning(
                    "%s: Failed to update branch; tracking branch gone upstream: %s",
                    self.orig_path,
                    branch,
                )
        elif not self.is_ancestor(branch, tracking_branch):
            logger.warning("%s: Failed to update branch; not fast-forward: %s", self.orig_path, branch)
        elif self.is_ancestor(tracking_branch, branch):
            logger.debug("%s: no changes for branch: %s", self.orig_path, branch)
        elif checked_out:
            # branch is checked out *somewhere*, but not necessarily here...
            if self.head.is_detached or self.head.ref != branch:
                # its checked out somewhere else!
                logger.info("%s: skipping branch checked out elsewhere: %s", self.orig_path, branch)
            elif self.is_dirty():
                logger.warning(
                    "%s: Refusing to update checked out branch: local changes would be overwritten: %s",
                    self.orig_path,
                    branch,
                )
            else:
                logger.info("%s: fast-forwarding checked out branch: %s", self.orig_path, branch)
                self.head.reset(tracking_branch, index=True, working_tree=True)
        else:
            branch.commit = tracking_branch.commit
            logger.info("%s: Updated branch: %s", self.orig_path, branch)

    def update(self: Self, lprune: bool = False) -> bool:
        failures = False
        for remote in self.remotes:
            logger.debug("%s: fetching: %s", self.orig_path, remote)
            start = time.monotonic()
            while True:
                fetchinfolist = []
                try:
                    fetchinfolist = remote.fetch()
                    break
                except git.GitCommandError as exc:
                    logger.exception("%s: %s", self.orig_path, exc)
                    if time.monotonic() - start > 120:
                        logger.error("%s: caught the thing too many times; giving up", self.orig_path)
                        failures = True
                        break
                    else:
                        logger.warning("%s: caught the thing; sleeping...", self.orig_path)
                        time.sleep(5)
                except Exception:
                    logger.exception("%s: Failed to fetch: %s", self.orig_path, remote)
                    failures = True
                    break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s: fetchinfolist: %s: %s", self.orig_path, remote, self.fetchinfolist_to_str(fetchinfolist)
                )

        for branch in self.branches:
            try:
                self.update_branch(branch, lprune)
            except Exception as exc:
                logger.exception(
                    "%s: Failed to update branch: %s (%s)",
                    self.orig_path,
                    branch,
                    exc,
                )
                failures = True

        return failures

    @cached_property
    def worktrees(self: Self) -> list[Self]:
        """
        given a git Repo, return a list of Repos representing all worktrees related
        to that Repo (including the passed Repo)
        """
        MyType = type(self)
        common_dir = Path(self.common_dir).resolve()
        parent_repo = MyType(common_dir.parent)
        known_repos = [parent_repo]
        worktrees_dir = common_dir.joinpath("worktrees")
        if worktrees_dir.exists():
            for child in worktrees_dir.iterdir():
                gitdir_file = child.joinpath("gitdir")
                gitdir_path = Path(gitdir_file.read_text().strip())
                child_worktree_repo = MyType(gitdir_path.parent)
                known_repos.append(child_worktree_repo)

        if self not in known_repos:
            # repo's haunted
            raise ValueError("passed repo not in list of known repos", self, known_repos)

        return known_repos

    def branch_is_checked_out_anywhere(self: Self, branch: git.Head) -> bool:
        """
        find out whether this branch is checked out in this worktree or any related
        worktree
        """

        for worktree in self.worktrees:
            # can't do a simple equality check on `branch`, b/c we may be from different Repos
            if (not worktree.head.is_detached) and worktree.head.ref.path == branch.path:
                # this branch is checked out in this worktree
                return True
        return False

    @classmethod
    def fetchinfo_to_str(cls: type[Self], fetchinfo: git.FetchInfo) -> str:
        out = str(fetchinfo.ref)
        if (remote_ref := fetchinfo.remote_ref_path) and (remote_ref := str(remote_ref).strip()):
            out = f"({remote_ref}) {out}"
        if fetchinfo.flags & (git.FetchInfo.FORCED_UPDATE | git.FetchInfo.FAST_FORWARD):
            out = f"{out} <- {fetchinfo.old_commit}"
        flag_list = [flag_name for flag, flag_name in FETCHINFO_FLAGS.items() if flag & fetchinfo.flags]
        flags = " | ".join(flag_list)
        out = f"{out} ({flags})"
        return out

    @classmethod
    def fetchinfolist_to_str(cls: type[Self], fetchinfolist: list[git.FetchInfo], indent=4, show_all=False) -> str:
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


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
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
    #     "-u", "--update", action="store_true", default=True, help="update repositories (the default)"
    # )
    # action_group.add_argument(
    #     "-s", "--status", action="store_true", help="report the status of located repositories"
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

    parser.add_argument(
        "-r",
        "--random-order",
        action="store_true",
        help="randomize the order in which repositories are accessed",
    )

    parser.add_argument(
        "-a",
        "--absolute",
        action="store_true",
        help="make repository paths absolute before operating on them",
    )

    volume_group = parser.add_mutually_exclusive_group()
    volume_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase the verbosity of the script (can be specified up to twice)",
    )
    volume_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="silence all output (cannot be combined with --verbose)",
    )

    traceback_group = parser.add_mutually_exclusive_group()
    traceback_group.add_argument(
        "-t",
        "--traceback",
        action="store_true",
        help="display tracebacks on error (defaults to True when verbose >= 2)",
    )
    traceback_group.add_argument(
        "-T",
        "--no-traceback",
        action="store_true",
        help="disable tracebacks (even when verbose >= 2)",
    )

    args = parser.parse_args(argv)

    # the `no_traceback_log_filter` logic is intentionally long-winded
    # in order to be more readable

    # disable traceback by default
    logger.addFilter(no_traceback_log_filter)

    if args.traceback:
        logger.removeFilter(no_traceback_log_filter)

    if args.quiet:
        logger.level = logging.FATAL
    elif args.verbose >= 2:
        logger.level = logging.DEBUG
        # re-enable tracebacks when verbose => 2
        logger.removeFilter(no_traceback_log_filter)
        # unless `--no-traceback` is specified
        if args.no_traceback:
            logger.addFilter(no_traceback_log_filter)
    elif args.verbose >= 1:
        logger.level = logging.INFO

    return args


def _main_singlethreaded(args: argparse.Namespace) -> bool:
    """update git repositories from remotes, one at a time."""
    failures = False

    for directory in args.directories:
        repos: list[Repo] = Repo.find_recursive(
            directory,
            maxdepth=args.depth,
            exclude=args.exclude,
            randomized=args.random_order,
            force_recurse=args.force_recurse,
            absolute_paths=args.absolute,
        )

        if args.print:
            for repo in repos:
                print(repo)
        else:
            for repo in repos:
                if args.slow:
                    time.sleep(1)
                try:
                    failures |= repo.update(args.lprune)
                except Exception as exc:
                    logger.exception("Trapped error: %s", exc)
                    failures = True

    return failures


def _main_multithreaded(args: argparse.Namespace) -> bool:
    executor = MyExecutor(os.cpu_count())
    failures = False

    for directory in args.directories:
        repos: list[Repo] = Repo.find_recursive(
            directory,
            maxdepth=args.depth,
            exclude=args.exclude,
            randomized=args.random_order,
            force_recurse=args.force_recurse,
            absolute_paths=args.absolute,
        )

        for repo in repos:
            executor.submit(repo.update, args.lprune)

    try:
        for ftr in as_completed(executor.futures()):
            try:
                failures |= ftr.result()
            except Exception as exc:
                logger.exception("Trapped error: %s", exc)
                failures = True
    except KeyboardInterrupt:
        for ftr in executor.futures():
            ftr.cancel()
    finally:
        executor.shutdown()

    return failures


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Update git repositories from remotes."""

    args = parse_arguments(argv)

    os.environ["GIT_ASKPASS"] = "echo"

    try:
        if args.single_thread or args.print or args.slow:
            failures = _main_singlethreaded(args)
        else:
            failures = _main_multithreaded(args)
    except KeyboardInterrupt:
        print("Caught interrupt")
        exit(0)

    exit(failures)


if __name__ == "__main__":
    main()
