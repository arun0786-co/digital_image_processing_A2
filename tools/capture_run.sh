#!/bin/zsh
# run a command in a new Terminal window from the repository root, wait for it
# to finish, then save a screenshot of just that window.
# usage: tools/capture_run.sh "<command>" results/problem_k/name.png [wait_seconds]
set -e
cmd="$1"
out="$2"
limit="${3:-1800}"
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$out" = /* ]] && target="$out" || target="$root/$out"
marker="$(mktemp -u /tmp/capture_run.XXXXXX)"

window_id=$(osascript <<EOF
tell application "Terminal"
    activate
    set run_tab to do script "cd '$root' && clear && $cmd; touch '$marker'"
    delay 0.5
    set run_window to front window
    if tty of selected tab of run_window is not tty of run_tab then error "front window is not the new tab"
    set bounds of run_window to {40, 40, 1240, 1000}
    return id of run_window
end tell
EOF
)
[[ "$window_id" == <-> ]] || { echo "no window id: $window_id" >&2; exit 1; }

waited=0
while [[ ! -e "$marker" ]]; do
    sleep 1
    waited=$((waited + 1))
    if (( waited > limit )); then
        echo "command did not finish within ${limit} s, window $window_id left open" >&2
        exit 1
    fi
done
rm -f "$marker"
sleep 1

mkdir -p "$(dirname "$target")"
screencapture -x -o -l "$window_id" "$target"
osascript -e "tell application \"Terminal\" to close (every window whose id is $window_id)" >/dev/null
echo "saved $target (window $window_id, ${waited} s)"
