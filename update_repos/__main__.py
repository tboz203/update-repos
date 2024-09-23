"""
Update git repositories from remotes.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional, ParamSpec

from .repo import Repo

logger = logging.getLogger(__package__)
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(levelname)8s %(asctime)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "verbose": {
                "format": "[%(levelname)8s %(asctime)s %(threadName)s %(name)s] %(message)s",
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
            __package__: {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }
)


class MyExecutor(ThreadPoolExecutor):
    """
    A subclass of `concurrent.futures.Executor` that keeps track of all futures
    it creates, and provides access to them via a generator method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._futures = []

    def submit(self, *args, **kwargs) -> Future:
        ftr = super().submit(*args, **kwargs)
        self._futures.append(ftr)
        return ftr

    def futures(self) -> Iterator[Future]:
        yield from self._futures


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
        logger.info("Caught interrupt")
        exit(1)

    exit(failures)


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


def _main_singlethreaded(args: argparse.Namespace) -> bool:
    """update git repositories from remotes, one at a time."""
    failures = False

    P = ParamSpec("P")

    def process(func: Callable[P, bool], *func_args: P.args, **func_kwargs: P.kwargs):
        nonlocal failures
        if args.slow:
            time.sleep(1)
        try:
            failures |= func(*func_args, **func_kwargs)
        except Exception as exc:
            logger.exception("Caught an exception: %s", exc)
            failures = True

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
                print(repo.orig_path)
        else:
            linked_worktrees: list[Repo] = []
            for repo in repos:
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


main()
