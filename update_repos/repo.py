"""
A customized and extended subclass of `git.Repo`
"""

from __future__ import annotations

import fnmatch
import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from typing import Any, Optional, Self, TypeAlias, Union

import git

# the triplets that `os.walk` & friends yield
WalkEntry: TypeAlias = tuple[str, list[str], list[str]]
# things that resolve to a commit
CommitRef: TypeAlias = Union[git.Commit, git.Tag, git.Head]

logger = logging.getLogger(__package__)

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


class Repo(git.Repo):
    """Just like git.Repo, but with more"""

    def __init__(
        self,
        pathlike,
        *args,
        is_main_worktree: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(pathlike, *args, **kwargs)
        self.orig_path = str(pathlike)
        self._is_main_worktree: Optional[bool] = is_main_worktree
        self._sibling_worktrees: Optional[list[Self]] = None

    @classmethod
    def find_recursive(
        cls,
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

    def update(self, lprune: bool = False) -> bool:
        logger.info("%s: Updating repository", self.orig_path)
        failures = False
        if self.is_main_worktree:
            failures |= self.update_remotes()
            failures |= self.update_branches(lprune)
        else:
            failures |= self.update_checked_out_branch()
        return failures

    def update_remotes(self) -> bool:
        failures = False
        for remote in self.remotes:
            if remote.config_reader.get("skipFetchAll", fallback=None):
                continue
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
                        logger.error(
                            "%s: caught the thing too many times; giving up",
                            self.orig_path,
                        )
                        failures = True
                        break
                    else:
                        logger.warning(
                            "%s: caught the thing; sleeping...", self.orig_path
                        )
                        time.sleep(5)
                except Exception:
                    logger.exception("%s: Failed to fetch: %s", self.orig_path, remote)
                    failures = True
                    break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s: fetchinfolist: %s: %s",
                    self.orig_path,
                    remote,
                    self.fetchinfolist_to_str(fetchinfolist),
                )
        return failures

    def update_branches(self, lprune: bool = False) -> bool:
        """update all local branches from their respective tracking branches (if set)."""
        failures = False
        for branch in self.branches:
            try:
                failures |= self.update_branch(branch, lprune)
            except Exception as exc:
                logger.exception(
                    "%s: Failed to update branch: %s (%s)", self.orig_path, branch, exc
                )
                failures = True

        return failures

    def update_branch(self, branch: git.Head, lprune: bool = False) -> bool:
        """fast-forward the given local branch from its tracking branch, if set."""
        failure = False
        checked_out = self.branch_is_checked_out_anywhere(branch)
        tracking_branch = branch.tracking_branch()
        if tracking_branch is None:
            logger.info("%s: no tracking branch set: %s", self.orig_path, branch)
        elif not tracking_branch.is_valid():
            if (not checked_out) and lprune:
                logger.info("%s: Pruning local branch: %s", self.orig_path, branch)
                self.delete_head(branch, force=True)
            else:
                failure = True
                logger.warning(
                    "%s: Failed to update branch; tracking branch gone upstream: %s",
                    self.orig_path,
                    branch,
                )
        elif self.is_ancestor(tracking_branch, branch):
            logger.debug("%s: no changes for branch: %s", self.orig_path, branch)
        elif not self.is_ancestor(branch, tracking_branch):
            failure = True
            logger.warning(
                "%s: Failed to update branch; not fast-forward: %s",
                self.orig_path,
                branch,
            )
        elif checked_out:
            # branch is checked out *somewhere*, but not necessarily here...
            if self.head.is_detached or self.head.ref != branch:
                # its checked out somewhere else!
                logger.info(
                    "%s: skipping branch checked out elsewhere: %s",
                    self.orig_path,
                    branch,
                )
            elif self.is_dirty():
                failure = True
                logger.warning(
                    "%s: Refusing to update dirty checked out branch: %s",
                    self.orig_path,
                    branch,
                )
            else:
                logger.info(
                    "%s: fast-forwarding checked out branch: %s", self.orig_path, branch
                )
                self.head.reset(tracking_branch, index=True, working_tree=True)
        else:
            branch.commit = tracking_branch.commit
            logger.info("%s: Updated branch: %s", self.orig_path, branch)
        return failure

    def update_checked_out_branch(self) -> bool:
        """update the currently checked out branch of this repository"""
        failure = False
        if self.head.is_detached:
            logger.info("%s: repo is in detached state", self.orig_path)
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

        MyType = type(self)
        worktrees = []
        for chunk in porcelain.split("\n\n"):
            parts = dict()
            for line in chunk.splitlines():
                lineparts = line.split(" ", maxsplit=1)
                if len(lineparts) == 1:
                    key, value = line, True
                elif len(lineparts) == 2:
                    key, value = lineparts
                else:
                    logger.warning("%s: lineparts: %s", lineparts)
                    continue
                parts[key] = value
            if parts["worktree"] == self.working_dir:
                worktrees.append(self)
                continue
            # the first worktree listed is the `main` one
            main_worktree = not worktrees
            worktree = MyType(parts["worktree"], is_main_worktree=main_worktree)
            worktrees.append(worktree)

        if self not in worktrees:
            raise RuntimeError("how'd that happen?", self, worktrees)

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
        if self._is_main_worktree is None:
            # the first worktree listed is the `main` one
            self._is_main_worktree = self is self.worktrees[0]
        return self._is_main_worktree

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
) -> Iterator[WalkEntry]:
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
