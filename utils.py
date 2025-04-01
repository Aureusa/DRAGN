def print_box(message: str):
    """
    Print a message in a box format.
    The box is created using Unicode box-drawing characters.

    :param message: The message to be printed in the box.
    :type message: str
    """
    lines = message.split('\n')
    max_length = max(len(line) for line in lines)
    border_up = '┌' + '─' * (max_length + 2) + '┐'
    border_down = '└' + '─' * (max_length + 2) + '┘'
    print(border_up)
    for line in lines:
        print(f'│ {line.ljust(max_length)} │')
    print(border_down)