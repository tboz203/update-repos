unset -f __update_repos_comp
__update_repos_comp () {

    # usage: update-repos [-h] [-d DEPTH] [-e EXCLUDE] [-R] [-1] [-p] [-l] [-s] [-r]
    #                     [-v | -q] [-t | --no-traceback]
    #                     [directories [directories ...]]

    # Update git repositories from remotes.

    # positional arguments:
    # directories           directories to search for repositories (defaults to
    #                         current directory)

    # optional arguments:
    # -h, --help            show this help message and exit
    # -d DEPTH, --depth DEPTH
    #                         maximum depth to search for repositories
    # -e EXCLUDE, --exclude EXCLUDE
    #                         directories to exclude from consideration; may be
    #                         repeated; accepts wildcards (remember to quote them!)
    # -R, --force-recurse   recurse into repositories, updating submodules &
    #                         simple nested repositories
    # -1, --single-thread   Run in a single thread. useful for background tasks,
    #                         or debugging.
    # -p, --print           don't update; print the paths of located repositories
    # -l, --lprune          If a local branch tracks a remote branch that no
    #                         longer exists, prune the local branch
    # -s, --slow            try to avoid ratelimit. implies `--single-thread`
    # -r, --random-order    randomize the order in which repositories are accessed
    # -v, --verbose         increase the verbosity of the script (can be specified
    #                         up to twice)
    # -q, --quiet           silence all output (cannot be combined with --verbose)
    # -t, --traceback       display tracebacks on error (defaults to True when
    #                         verbose >= 2)
    # --no-traceback        disable tracebacks (even when verbose >= 2)

    local cur prev words cword
    _init_completion || return

    # handle options expecting arguments
    case "$prev" in
        (-e|--exclude)
            # --exclude expects a directory pattern, we'll just offer directories
            _filedir -d
            return ;;
        (-d|--depth)
            # --depth expects a number, but listing out all possible numbers would take too long
            COMPREPLY=("enter a number..." "")
            return ;;
    esac


    # the full battery of options
    optional=(
        -h -d -e -R -1 -p -l -s -r -v -q -t --help --depth --exclude --force-recurse --single-thread --print --lprune
        --slow --random-order --verbose --quiet --traceback --no-traceback
    )

    # only included if you're already typing an option
    if [[ $cur == -* ]]; then
        COMPREPLY=( $(compgen -W "${optional[*]}" -- "$cur") )
    fi

    # plus directories
    _filedir -d
}

# the `nosort` option appears to have been added in bash 4.4
if [[ ${BASH_VERSINFO[0]} -ge 5 || ( ${BASH_VERSINFO[0]} -eq 4 && ${BASH_VERSINFO[1]} -ge 4 ) ]]; then
    complete -o nosort -F __update_repos_comp update-repos
    complete -o nosort -F __update_repos_comp update_repos.py
else
    complete -F __update_repos_comp update-repos
    complete -F __update_repos_comp update_repos.py
fi
