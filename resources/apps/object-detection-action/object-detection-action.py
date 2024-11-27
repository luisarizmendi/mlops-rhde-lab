import os
import time
import requests


def get_detections(endpoint: str) -> dict:
    """
    Fetch detections from the endpoint and return them as a dictionary.
    """
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        # Parse the JSON response
        detections = response.json()

        # Ensure the response is a dictionary
        if isinstance(detections, dict):
            return detections
        else:
            print(f"Unexpected response format: {detections}")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching detections: {e}")
        return {}
    except ValueError as e:
        print(f"Error parsing JSON from {endpoint}: {e}")
        return {}


def send_alert(alert_endpoint: str, message: str) -> None:
    """
    Send an alert message to the specified endpoint.
    """
    try:
        response = requests.post(alert_endpoint, json={"message": message}, timeout=5)
        response.raise_for_status()
        print(f"Sent alert: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")


def send_alive_signal(alive_endpoint: str) -> None:
    """
    Send an alive signal to the specified endpoint.
    """
    try:
        response = requests.post(alive_endpoint, json={"status": "alive"}, timeout=5)
        response.raise_for_status()
        print("Sent alive signal")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alive signal: {e}")


def main():
    # Configuration from environment variables
    detections_endpoint = os.getenv("DETECTIONS_ENDPOINT", "http://localhost:5000/current_detections")
    alert_endpoint = os.getenv("ALERT_ENDPOINT", "http://localhost:6000/alert")
    alive_endpoint = os.getenv("ALIVE_ENDPOINT", "http://localhost:6000/alive")
    check_interval = float(os.getenv("CHECK_INTERVAL", 1))  # Seconds
    alert_duration = float(os.getenv("ALERT_DURATION", 5))  # Seconds
    reset_checks = int(os.getenv("RESET_CHECKS", 3))  # Number of checks
    alive_interval = float(os.getenv("ALIVE_INTERVAL", 5))  # Seconds

    last_alert_time = 0
    alert_active = False
    reset_count = 0
    last_alive_signal = time.time()

    while True:
        detections = get_detections(detections_endpoint)

        # Check if "no_helmet" or "no_shoes" exists with a count > 0
        has_no_helmet = detections.get("no_helmet", {}).get("count", 0) > 0
        has_no_shoes = detections.get("no_shoes", {}).get("count", 0) > 0

        if has_no_helmet or has_no_shoes:
            if not alert_active:
                last_alert_time = time.time()
                print(f"Detected: {'no_helmet' if has_no_helmet else 'no_shoes'}")

            if time.time() - last_alert_time >= alert_duration:
                if not alert_active:
                    send_alert(alert_endpoint, "ALERT_ON")
                    alert_active = True
                reset_count = 0
        else:
            if alert_active:
                reset_count += 1
                if reset_count >= reset_checks:
                    send_alert(alert_endpoint, "ALERT_OFF")
                    alert_active = False
                    reset_count = 0
            else:
                last_alert_time = time.time()

        # Send alive signal
        if time.time() - last_alive_signal >= alive_interval:
            send_alive_signal(alive_endpoint)
            last_alive_signal = time.time()

        time.sleep(check_interval)


if __name__ == "__main__":
    main()
