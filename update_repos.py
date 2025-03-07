#!/usr/bin/env python
"""Update git repositories from remotes."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import logging
import logging.config
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import git

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from concurrent.futures import Future

    # the triplets that `os.walk` & friends yield
    type WalkEntry[T: (str, Path)] = tuple[T, list[str], list[str]]

    # things that resolve to a commit
    type CommitRef = git.Commit | git.Tag | git.Head

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


@dataclass(frozen=True)
class Settings:
    """A dataclass to encapsulate script settings."""

    directories: tuple[str, ...]
    exclude: tuple[str, ...]

    depth: int | None
    verbosity: int

    absolute: bool
    follow: bool
    lprune: bool
    print: bool
    recurse: bool
    single_thread: bool
    search_hidden: bool
    search_caches: bool
    slow: bool
    log_tracebacks: bool

    order: Literal["random", "alphabetical"] | None


class ContextAddedLogFilter(logging.Filter):
    """A logging filter that populates a LogRecord attribute from a ContextVar."""

    def __init__(self, *, var: ContextVar, attribute: str):
        self.var = var
        self.attribute = attribute

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, self.attribute, self.var.get())
        return True


class NoTracebackLogFilter(logging.Filter):
    """A log filter that unconditionally removes exception info."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Remove exception info from a log record"""
        record.exc_info = None

        return True


class MyExecutor(ThreadPoolExecutor):
    """A subclass of `ThreadPoolExecutor` that keeps track of all futures it
    creates, and provides access to them via a generator method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._futures = []

    def submit[**P, T](
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        ftr = super().submit(fn, *args, **kwargs)
        self._futures.append(ftr)
        return ftr

    @property
    def futures(self) -> Generator[Future]:
        yield from self._futures


class Repo(git.Repo):
    """Just like git.Repo, but with more"""

    def __init__(self, path: str | Path, *args, **kwargs):
        super().__init__(Path(path).resolve(), *args, **kwargs)
        self.orig_path = str(path)
        self._worktrees: list[Self] | None = None

    @classmethod
    def find_recursive(
        cls,
        starting_path: str | Path,
        *,
        max_depth: int | None = None,
        exclude: Iterable[str] | None = None,
        include_ancestors: bool = True,
        recurse_repositories: bool = False,
        absolute_paths: bool = False,
        follow_symlinks: bool = True,
        search_hidden: bool = False,
        search_caches: bool = False,
    ) -> Generator[Self]:
        """
        Enumerate Git Repositories.

        Search first in and beneath the given `path` (optionally limited to a
        maximum depth), and then optionally also upwards through the path's
        ancestors. Skip searching paths matching optional wildcard exclusion
        patterns. Do not recurse into found repositories unless
        `recurse_repositories` is True. Build enumerated repositories from
        relative paths unless `absolute_paths` is True.
        """

        starting_path = Path(starting_path)
        if absolute_paths:
            starting_path = starting_path.resolve()

        logger.debug(
            "%s: finding git repositories: (maxdepth=%s)", starting_path, max_depth
        )

        walk_generator = depthwalk(
            starting_path, max_depth=max_depth, follow_symlinks=follow_symlinks
        )
        if include_ancestors:
            walk_generator = chain(walk_generator, backwalk(starting_path))

        exclude = tuple(exclude) if exclude else ()
        exclude += () if search_hidden else (".*",)
        exclusion_patterns = cls._build_exclusion_patterns(exclude)

        def should_exclude(
            path: str | Path, dirs: Iterable[str | Path], files: Iterable[str | Path]
        ) -> bool:
            """Test whether or not the current path should be excluded from the walk."""

            del dirs, files

            if any(fnmatch.fnmatch(str(path), pat) for pat in exclusion_patterns):
                return True

            if not search_caches and is_cache(path):
                return True

            return False

        for path, dirs, files in walk_generator:
            if should_exclude(path, dirs, files):
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
                if not recurse_repositories:
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
                exclusions.add(pattern)
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
        """Fetch from all remotes, and return a bool indicating failures."""
        with context(REPO_DIR, self.orig_path):
            failures = False
            for remote in self.remotes:
                if remote.config_reader.get("skipFetchAll", fallback=None):
                    logger.debug("skipping: %s", remote)
                    continue
                logger.debug("fetching: %s", remote)
                try:
                    fetchinfolist = remote.fetch()
                except git.GitCommandError as exc:
                    logger.error(exc)
                    failures = True
                except Exception:
                    logger.exception("Failed to fetch: %s", remote)
                    failures = True
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "fetchinfolist: %s: %s",
                            remote,
                            self.fetchinfolist_to_str(fetchinfolist),
                        )
            return failures

    def _update_remotes(self) -> bool:
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
                        logger.error(exc)
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
                logger.debug("no tracking branch set: %s", branch)
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
                    logger.debug("fast-forwarding checked out branch: %s", branch)
                    self.head.reset(tracking_branch, index=True, working_tree=True)
            else:
                branch.commit = tracking_branch.commit
                logger.debug("Updated branch: %s", branch)
            return failure

    def update_checked_out_branch(self) -> bool:
        """update the currently checked out branch of this repository"""
        with context(REPO_DIR, self.orig_path):
            failure = False
            if self.head.is_detached:
                logger.info("Skipping repo in detached state")
            elif not isinstance(self.head.ref, git.Head):
                logger.warning("%s: HEAD not pointed at a Head?: %s", self.head.ref)
            else:
                failure = self.update_branch(self.head.ref)
            return failure

    def _find_worktrees(self) -> list[Self]:
        """Collect the Repos of worktrees related to this Repo."""

        porcelain: str = self.git.worktree("list", "--porcelain")
        raw_worktrees = _parse_porcelain_key_value_stanzas(porcelain)

        worktrees = []
        RepoType = type(self)
        for worktree_values in raw_worktrees:
            worktree_dir = worktree_values.get("worktree")
            if not worktree_dir:
                logger.warning("Malformed worktree: %s", worktree_values)
            elif worktree_dir == self.working_dir:
                worktrees.append(self)
            else:
                assert isinstance(worktree_dir, str)
                worktree = RepoType(worktree_dir)
                worktrees.append(worktree)

        return worktrees

    @property
    def worktrees(self) -> list[Self]:
        """The list of Repos for all worktrees related to this Repo (including this one)."""
        if self._worktrees is not None:
            return self._worktrees

        porcelain: str = self.git.worktree("list", "--porcelain")
        raw_worktrees = _parse_porcelain_key_value_stanzas(porcelain)

        worktrees = []
        RepoType = type(self)
        for worktree_values in raw_worktrees:
            worktree_dir = worktree_values.get("worktree")
            if not worktree_dir:
                logger.warning("Malformed worktree: %s", worktree_values)
            elif worktree_dir == self.working_dir:
                worktrees.append(self)
            else:
                assert isinstance(worktree_dir, str)
                worktree = RepoType(worktree_dir)
                worktrees.append(worktree)

        if self not in worktrees:
            raise RuntimeError(
                "Own worktree not found in worktrees list", self, porcelain
            )

        for worktree in worktrees:
            worktree.worktrees = worktrees

        return worktrees

    @worktrees.setter
    def worktrees(self, worktrees: list[Self]):
        if self not in worktrees:
            raise ValueError(
                "Repo must be among its own sibling worktrees", self, worktrees
            )
        self._worktrees = list(worktrees)

    @property
    def is_main_worktree(self) -> bool:
        return self.git_dir == self.common_dir

    def branch_is_checked_out_anywhere(self, branch: git.Head) -> bool:
        """
        Find out whether this branch is checked out in this worktree or any related
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
        flag_list = [
            flag_name
            for flag, flag_name in FETCHINFO_FLAGS.items()
            if flag & fetchinfo.flags
        ]
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
    starting_path: str | Path,
    onerror: Callable[[OSError], object] | None = None,
    top_down: bool = True,
    follow_symlinks: bool = True,
    max_depth: int | None = None,
) -> Generator[WalkEntry[Path]]:
    """just like `os.walk`, but with new added `maxdepth` parameter!"""
    starting_path = Path(starting_path)
    walk_generator = starting_path.walk(
        top_down=top_down, on_error=onerror, follow_symlinks=follow_symlinks
    )
    if max_depth is None:
        yield from walk_generator
        return

    depthmap = {starting_path: 0}
    for path, dirs, files in walk_generator:
        yield path, dirs, files
        here = depthmap[path]
        if here >= max_depth:
            dirs.clear()
            continue
        for dir in dirs:
            depthmap[path / dir] = here + 1


def backwalk(starting_path: str | Path) -> Generator[WalkEntry[Path]]:
    """similar to `os.walk`, but goes up instead of down"""
    for path in Path(starting_path).parents:
        dirs, files = [], []
        for item in path.iterdir():
            place = dirs if item.is_dir() else files
            place.append(item.name)

        yield path, dirs, files


def is_cache(path: str | Path) -> bool:
    """Decide whether a given path is a cache directory."""
    path = Path(path)
    if re.match(r"^(\W|_)*cache(\W|_)*$", path.name, re.IGNORECASE):
        return True

    cachedir_tag = path / "CACHEDIR.TAG"
    if cachedir_tag.is_file(follow_symlinks=False):
        signature = b"Signature: 8a477f597d28d172789f06886806bc55"
        with cachedir_tag.open("rb") as fin:
            header = fin.read(len(signature))
        if header == signature:
            return True

    return False


@contextmanager
def context[T](var: ContextVar[T], value: T):
    """Set a context variable within a context block."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def _parse_porcelain_key_value_stanzas(
    porcelain_text: str,
) -> list[dict[str, str | bool]]:
    """Parse porcelain command key/value stanzas into a list of dicts."""
    # keeping track of the current stanza and the list of all stanzas, we go
    # line by line through our input collecting key value pairs

    current_stanza = {}
    stanzas = [current_stanza]

    for line in porcelain_text.splitlines():
        # stanzas are separated by blank lines
        if not line:
            current_stanza = {}
            stanzas.append(current_stanza)
            continue
        # stanza lines are either key value pairs, or bare keys
        lineparts = line.split(" ", maxsplit=1)
        if len(lineparts) == 2:
            key, value = lineparts
        else:
            key, value = line, True
        current_stanza[key] = value

    return stanzas


def parse_arguments(argv: Sequence[str] | None) -> Settings:
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
        "-E",
        "--everywhere",
        action="count",
        default=0,
        help="don't skip hidden directories; if given twice, also don't skip caches",
    )
    parser.add_argument(
        "-R",
        "--recurse-repositories",
        action=argparse.BooleanOptionalAction,
        help="recurse into repositories, updating submodules & simple nested repositories",
    )
    parser.add_argument(
        "-1",
        "--single-thread",
        action=argparse.BooleanOptionalAction,
        help="Run in a single thread. useful for background tasks, or debugging.",
    )
    parser.add_argument(
        "-p",
        "--print",
        action=argparse.BooleanOptionalAction,
        help="Don't update anything; just print the paths of located repositories",
    )
    parser.add_argument(
        "-l",
        "--lprune",
        action=argparse.BooleanOptionalAction,
        help="Prune local branches tracking remote branches that no longer exist",
    )
    parser.add_argument(
        "-s",
        "--slow",
        action=argparse.BooleanOptionalAction,
        help="Try to avoid ratelimit. Implies `--single-thread`",
    )
    parser.add_argument(
        "-a",
        "--absolute",
        action=argparse.BooleanOptionalAction,
        help="Make repository paths absolute before operating on them",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the script",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease the verbosity of the script",
    )
    parser.add_argument(
        "-L",
        "--follow",
        "--dereference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow symbolic links",
    )
    parser.add_argument(
        "-t",
        "--tracebacks",
        action=argparse.BooleanOptionalAction,
        help="Display tracebacks on error (defaults to True when verbosity >= 2)",
    )

    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument(
        "-r",
        "--random-order",
        dest="order",
        action="store_const",
        const="random",
        default=None,
        help="Operate on repositories in a randomized order",
    )
    order_group.add_argument(
        "-S",
        "--sorted",
        dest="order",
        action="store_const",
        const="alphabetical",
        default=None,
        help="Operate on repositories in alphabetical order",
    )

    args = parser.parse_args(argv)

    verbosity: int = args.verbose - args.quiet
    log_tracebacks: bool = (
        args.tracebacks if args.tracebacks is not None else (verbosity >= 2)
    )

    return Settings(
        directories=tuple(args.directories),
        exclude=tuple(args.exclude) if args.exclude else (),
        depth=args.depth,
        verbosity=verbosity,
        absolute=bool(args.absolute),
        follow=bool(args.follow),
        lprune=bool(args.lprune),
        print=bool(args.print),
        recurse=bool(args.recurse_repositories),
        single_thread=bool(args.single_thread) or bool(args.slow),
        search_hidden=(args.everywhere >= 1),
        search_caches=(args.everywhere >= 2),
        slow=bool(args.slow),
        log_tracebacks=log_tracebacks,
        order=args.order,
    )


def setup_logging(settings: Settings) -> None:
    level = logging.WARNING - (settings.verbosity * 10)
    level = min(max(logging.DEBUG, level), logging.FATAL)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(levelname)8s %(asctime)s %(name)s] %(repo_dir)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "filters": {
                "repo_dir_context": {
                    "()": ContextAddedLogFilter,
                    "var": REPO_DIR,
                    "attribute": "repo_dir",
                },
                "no_tracebacks": {
                    "()": NoTracebackLogFilter,
                },
            },
            "handlers": {
                "console": {
                    "formatter": "standard",
                    "filters": ["repo_dir_context"]
                    + ([] if settings.log_tracebacks else ["no_tracebacks"]),
                    "class": "logging.StreamHandler",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )

    # if log_tracebacks is False or (log_tracebacks is None and volume <= 1):
    #     logger.addFilter(NoTracebackLogFilter())


def collect_repos(settings: Settings) -> Generator[Repo]:
    if settings.order == "random":
        settings = copy.replace(settings, order=None)
        repos = list(collect_repos(settings))
        random.shuffle(repos)
        yield from repos
        return

    seen: set[Repo] = set()
    for directory in settings.directories:
        for repo in Repo.find_recursive(
            starting_path=directory,
            max_depth=settings.depth,
            exclude=settings.exclude,
            include_ancestors=True,
            recurse_repositories=settings.recurse,
            absolute_paths=settings.absolute,
            follow_symlinks=settings.follow,
            search_hidden=settings.search_hidden,
            search_caches=settings.search_caches,
        ):
            if repo in seen:
                continue
            seen.add(repo)
            yield repo


def _single_threaded_main(settings: Settings) -> bool:
    """
    Update git repositories from remotes, one at a time.
    Returns a boolean indicating whether errors occurred.
    """

    any_errors = False

    def process[**P](
        func: Callable[P, bool], *func_args: P.args, **func_kwargs: P.kwargs
    ):
        if settings.slow:
            time.sleep(1)
        try:
            return func(*func_args, **func_kwargs)
        except Exception as exc:
            logger.exception("Caught an exception: %s", exc)
            return True

    linked_worktrees: list[Repo] = []
    for repo in collect_repos(settings):
        if not repo.is_main_worktree:
            linked_worktrees.append(repo)
            continue
        any_errors |= process(repo.update, lprune=settings.lprune)

    for repo in linked_worktrees:
        any_errors |= process(repo.update, lprune=settings.lprune)

    return any_errors


def _multi_threaded_main(settings: Settings) -> bool:
    """update git repositories from remotes in parallel."""
    any_errors = False
    linked_worktrees = []

    with MyExecutor() as executor:
        for repo in collect_repos(settings):
            if repo.is_main_worktree:
                executor.submit(repo.update, lprune=settings.lprune)
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

    settings = parse_arguments(argv)
    setup_logging(settings)

    logger.debug("Settings: %r", settings)

    try:
        if settings.print:
            for repo in collect_repos(settings):
                print(repo.orig_path)
            exit(0)

        if settings.single_thread:
            failures = _single_threaded_main(settings)
        else:
            failures = _multi_threaded_main(settings)
    except KeyboardInterrupt:
        logger.fatal("Caught interrupt")
        exit(1)
    except Exception as exc:
        logger.fatal("Fatal error: %s", exc, exc_info=True)
        exit(1)

    exit(failures)


if __name__ == "__main__":
    main()
