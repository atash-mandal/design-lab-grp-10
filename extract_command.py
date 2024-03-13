def extract_attributes(text):
    # Split the text into words
    words = text.split()

    # Initialize attributes
    action = None
    device = None

    # Define keywords for actions and devices
    action_keywords = {"turn": ["on", "off", "open", "close", "add", "authenticate"]}
    device_keywords = {"light": ["light1", "light2", "light3"], "fan": ["fan1", "fan2", "fan3", "fan4"]}

    # Search for action
    for word in words:
        if word in action_keywords:
            action = word
            break

    # Search for device
    for word in words:
        for key, values in device_keywords.items():
            if word in values:
                device = word
                break

    return action, device
