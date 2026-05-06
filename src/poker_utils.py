def card_rank(card):
    return str(card).strip()[0]


def card_suit(card):
    return str(card).strip()[-1]


def is_pair(card1, card2):
    return card_rank(card1) == card_rank(card2)


def is_suited(card1, card2):
    return card_suit(card1) == card_suit(card2)
