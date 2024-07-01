def calculate_cheat_score(output):
    cheat_score = 0

    # Identity check
    if 'Identity' in output and output['Identity'] is not None:
        if not output['Identity']:
            cheat_score += 50
    else:
        cheat_score += 20  # Assume the worst case if Identity is missing or None

    # Number of people check
    if 'Number of people' in output and output['Number of people'] is not None:
        num_people = int(output['Number of people'])
        if num_people != 1:
            cheat_score += 20
    else:
        cheat_score += 20  # Assume the worst case if Number of people is missing or None

    # Prohibited item use check
    if 'Prohibited Item Use' in output and output['Prohibited Item Use'] is not None:
        if output['Prohibited Item Use']:
            cheat_score += 20

    # Distance from prohibited item
    if 'Distance' in output and output['Distance'] is not None:
        distance = int(output['Distance'])
        if distance < 100:
            cheat_score += (70 - distance)

    # Face direction check
    if 'Face Direction' in output and output['Face Direction'] is not None:
        if output['Face Direction'] != 'forward':
            cheat_score += 5

    # Face zone check
    if 'Face Zone' in output and output['Face Zone'] is not None:
        if output['Face Zone'] == 'red':
            cheat_score += 20
        if output['Face Zone'] == 'yellow':
            cheat_score += 15
    else:
        cheat_score += 10

    # Eye direction check
    if 'Eye Direction' in output and output['Eye Direction'] is not None:
        if output['Eye Direction'] != 'center':
            cheat_score += 5

    # Mouth check
    if 'Mouth' in output and output['Mouth'] is not None:
        if output['Mouth'] != 'GREEN':
            cheat_score += 10

    # Normalize the score to 100
    cheat_score = min(cheat_score, 100)

    return cheat_score
