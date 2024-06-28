# Kill all tmux sessions created by tmux.py

# Get a list of all tmux sessions
sessions=$(tmux list-sessions -F "#{session_name}")

# Loop through each session name
for session in $sessions; {
  # Check if the session name starts with "graphtrail"
  if [[ $session == graphtrail* ]]; then
    # Kill the session
    tmux kill-session -t $session
  fi
}
