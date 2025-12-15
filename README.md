# Circles Object Size Meter

A program designed to detect spherical objects, measure their size, and display the resulting image along with a size distribution graph  on a dedicated 3.5" TFT screen on a Raspberry Pi 5.

## Project Details
- **Hardware:** Raspberry Pi 5, 3.5-inch TFT Display
- **Language:** Python 3.13.5
- **Features:**
  - Circle detection from image input.
  - Size calculation and distribution graph generation.
  - GUI output displayed on the TFT screen.
  - Automatic program execution upon system boot (Auto-start Service).

---

## Service Installation (Auto-start)

To ensure the program runs automatically every time the machine boots, we use `systemd`.

### 1. Create the Service File
Create the configuration file at `/etc/systemd/system/size-meter.service`:

```bash
sudo nano /etc/systemd/system/size-meter.service
```

 - #### Ini, TOML
    ```bash
    [Unit]
    Description=Size Meter Service
    # Start after the network and filesystem are ready
    After=network.target filesystem.target

    [Service]
    User=root
    Group=root

    # The directory containing the code
    WorkingDirectory=/home/circles/size-meter

    # Execute Python from the Virtual Environment
    ExecStart=/home/circles/size-meter/venv/bin/python3 /home/circles/size-meter/main.py

    # Always restart if the program fails (Auto-restart)
    Restart=always
    # Wait 3 seconds before restarting
    RestartSec=3

    # Send output (print) to the system log for debugging
    StandardOutput=inherit
    StandardError=inherit

    [Install]
    WantedBy=multi-user.target
    ```

### 2. Start the Service

```bash
# Reload the systemd configuration
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable size-meter.service

# Start the service immediately
sudo systemctl start size-meter.service
```

## Service Management (Control & Monitoring)

- ### Check Status

    ```bash
    sudo systemctl status size-meter.service
    ```

- ### Stopping and Starting the Service

    ```bash
    sudo systemctl stop size-meter.service
    ```

- ### Start the service (If it was previously stopped)

    ```bash
    sudo systemctl start size-meter.service
    ```
- ### Restart the service (Useful after code changes)

    ```bash
    sudo systemctl restart size-meter.service
    ```

- ### Permanently Disable the service (Stop it immediately and prevent it from running on subsequent boots) 

    ```bash
    sudo systemctl disable --now size-meter.service
    ```

## Viewing Logs (Troubleshooting)

If the program fails, or you need to see the values printed by print(), use the following commands

- ### View Logs in Real-time

    ```bash
    sudo journalctl -u size-meter.service -f
    ```

- ### View All Historical Logs

    ```bash
    sudo journalctl -u size-meter.service
    ```