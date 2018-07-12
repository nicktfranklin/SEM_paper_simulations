def parser_poetry(node):
    if node[0] == "BEGIN":
        return tuple(["Enter", node[1], None])
    if node[0] == "Order_drink":
        return tuple(["Order", node[1], node[2]])
    if node[0] == "Too_expensive":
        return tuple(["Cancel", node[1], None])
    if node[0] == "Sit_down":
        return tuple(["Sit/Greet", node[1], node[2]])
    if node[0] == "Emcee_intro":
        return tuple(["Introduce", node[1], node[2]])
    if node[0] == "Poet_performs":
        return tuple(["Perform", node[1], None])
    if node[0] == "Subject_declines":
        return tuple(["Decline", node[1], None])
    if node[0] == "Subject_performs":
        return tuple(["Decline", node[1], None])
    if node[0] == "Subject_performs":
        return tuple(["Decline", node[1], None])
    if node[0] == "Say_goodbye":
        return tuple(["Goodbye", node[1], node[2]])
    if node[0] == "Order_dessert":
        return tuple(["Order", node[1], node[2]])
    if node[0] == "END":
        return tuple(["Leave", node[1], None])


def parser_fight(node):
    if node[0] == "BEGIN":
        return tuple(["Enter", node[1], None])
    if node[0] == "Walk_to_front":
        return tuple(["Walk_to_front", node[1], node[2]])
    if node[0] == "Walk_to_back":
        return tuple(["Walk_to_back", node[1], node[2]])
    if node[0] == "Step_in_front":
        return tuple(["Step_in_front", node[1], node[2]])
    if node[0] == "Say_excuse_me":
        return tuple(["Say_excuse_me", node[1], None])
    if node[0] == "Ignore":
        return tuple(["Ignore", node[1], node[2]])
    if node[0] == "Shove":
        return tuple(["Shove", node[1], node[2]])
    if node[0] == "Subject_stares":
        return tuple(["Subject_stares", node[1], node[2]])
    if node[0] == "Turn_to_barista":
        return tuple(["Turn_to_barista", node[1], node[2]])
    if node[0] == "X_stare":
        return tuple(["X_stare", node[1], node[2]])
    if node[0] == "X_shove":
        return tuple(["X_shove", node[1], node[2]])
    if node[0] == "Cream_splash":
        return tuple(["Cream_splash", node[1], node[2]])
    if node[0] == "Dessert_crumble":
        return tuple(["Dessert_crumble", node[1], node[2]])
    if node[0] == "Call_policeman":
        return tuple(["Call_policeman", node[1], node[2]])
    if node[0] == "Barista_orders":
        return tuple(["Barista_orders", node[1], None])
    if node[0] == "Love_juice":
        return tuple(["Love_juice", node[1], node[2]])
    if node[0] == "Hate_coffee":
        return tuple(["Hate_coffee", node[1], node[2]])
    if node[0] == "END":
        return tuple(["HandNapkin", node[1], node[2]])


def parse_story(story, parser):
    story = story[:-2]  # remove line end symbol
    story = story.translate(None, ''.join('.'))  # remove the period at the end

    story = story.split('; ')  # separate the nodes

    # parse the nodes!
    parsed_story = list()
    for node in story:
        # remove the symbol punctuation -- note: order is sufficient here
        n = node.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        # what is the first word? The first word is really important for parsing
        parsed_story.append(parser(n))
    return parsed_story