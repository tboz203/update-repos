# mapfile -t __update_repos_options < <(
#     _parse_usage update-repos
#     _parse_help update-repos
# )

unset -f __update_repos_comp
__update_repos_comp() {
    : 'bash completion function for update-repos'
    # usage: update-repos [-h] [-d DEPTH] [-e EXCLUDE] [-R] [-1] [-p] [-l] [-s] [-r] [-a] [-v | -q] [-t | -T] [directories ...]
    #
    # Update git repositories from remotes.
    #
    # positional arguments:
    #   directories           directories to search for repositories (defaults to current directory)
    #
    # options:
    #   -h, --help            show this help message and exit
    #   -d DEPTH, --depth DEPTH
    #                         maximum depth to search for repositories
    #   -e EXCLUDE, --exclude EXCLUDE
    #                         directories to exclude from consideration; may be repeated; accepts wildcards (remember to quote them!)
    #   -R, --force-recurse   recurse into repositories, updating submodules & simple nested repositories
    #   -1, --single-thread   Run in a single thread. useful for background tasks, or debugging.
    #   -p, --print           don't update; print the paths of located repositories
    #   -l, --lprune          If a local branch tracks a remote branch that no longer exists, prune the local branch
    #   -s, --slow            try to avoid ratelimit. implies `--single-thread`
    #   -r, --random-order    randomize the order in which repositories are accessed
    #   -a, --absolute        make repository paths absolute before operating on them
    #   -v, --verbose         increase the verbosity of the script (can be specified up to twice)
    #   -q, --quiet           silence all output (cannot be combined with --verbose)
    #   -t, --traceback       display tracebacks on error (defaults to True when verbose >= 2)
    #   -T, --no-traceback    disable tracebacks (even when verbose >= 2)

    local cur prev split
    _init_completion -s || return

    # handle options expecting arguments
    case "$prev" in
    -e | --exclude)
        # --exclude expects a directory pattern, we'll just offer directories
        _filedir -d
        return
        ;;
    -d | --depth)
        # --depth expects a number, but listing out all possible numbers would take too long
        COMPREPLY=("enter a number..." "")
        return
        ;;
    esac

    # if we've split a `--word=arg` token, we ought to be done now
    $split && return

    # the full battery of options
    # (had been using _parse_usage & _parse_help, but that's slow...)
    local optional=(
        -d= -e=
        -h -R -1 -p -l -s -r -a -v -q -t -T
        --depth= --exclude=
        --help --force-recurse --single-thread --print --lprune --slow
        --random-order --absolute --verbose --quiet --traceback --no-traceback
    )

    # only include options if you're already typing an option
    if [[ $cur == -* ]]; then
        # mapfile -t COMPREPLY < <(compgen -W "${__update_repos_options[*]}" -- "$cur")
        mapfile -t COMPREPLY < <(compgen -W "${optional[*]}" -- "$cur")
        [[ ${COMPREPLY-} == *= ]] && compopt -o nospace
    fi

    # always include directories
    _filedir -d
} || return 1

# the `nosort` option appears to have been added in bash 4.4
if ((BASH_VERSINFO[0] >= 5 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4))); then
    complete -o nosort -F __update_repos_comp update-repos
else
    complete -F __update_repos_comp update-repos
fi
