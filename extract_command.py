def extract_attributes(text):
    words = text.split()

    # Initialize attributes
    action = None
    device = None

    # Define keywords for actions and devices
    action_keywords = {"turn": ["on", "off", "open", "close", "add", "authenticate"]}
    device_keywords = {"light": ["light"], "fan": ["fan"]}

    # Search for action
    for word in words:
        if word in action_keywords:
            action = word
            break

    # Search for device
    for word in words:
        for key, values in device_keywords.items():
            if word in values:
                device = key
                break

    return action, device
