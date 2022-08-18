# update_repos.py

A script to find and update git repositories in a directory hierarchy. By
default, walks the current directory to find git repositories (without
searching for nested repositories), fetches all remotes, and attempts to
update all local branches that have tracked branches.
