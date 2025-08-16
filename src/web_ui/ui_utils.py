import string
import types
import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import owlready2 as owl


def get_colors_from_palette(number):
    # Get the color palette from matplotlib
    if number <= 10:
        palette_name = 'tab10'
    elif number <= 20:
        palette_name = 'tab20'
    else:
        palette_name = 'viridis'

    try:
        palette = plt.get_cmap(palette_name)
    except ValueError:
        return f"Error: Palette '{palette_name}' not found."

    # Generate a list of colors from the palette
    if number <= 20:
        colors = [palette(i % palette.N) for i in range(number)]
    else:
        colors = [palette(i / number) for i in range(number)]

    # Convert the colors to their hexadecimal representation
    color_hex = [mcolors.rgb2hex(color[:3]) for color in colors]

    return color_hex


def get_closed_arguments(data) -> list[str]:
    unique_name = "".join(random.choices(string.ascii_letters, k=10))
    onto = owl.get_ontology(f'{unique_name}.owl')
    with onto:

        # Asserting attacks relation
        class attacks(owl.ObjectProperty):
            pass

        class isAttackedBy(owl.ObjectProperty):
            inverse_property = attacks

        class round(owl.AnnotationProperty, owl.FunctionalProperty):
            pass

        class text(owl.AnnotationProperty, owl.FunctionalProperty):
            pass

        class number(owl.AnnotationProperty, owl.FunctionalProperty):
            pass

        # Creating argument sets
        argument_class_mapping = {}
        argument_sets = data['argument_sets']
        for argument_set, arguments in argument_sets.items():
            Cl = types.new_class(argument_set, (owl.Thing,))
            Cl.label = argument_set
            for argument, text in arguments.items():
                inst = Cl()
                inst.label = argument
                argument_class_mapping[argument] = argument_set
                argument = argument.split('.')
                inst.text = text
                inst.round = argument[1]
                if len(argument) == 4:
                    inst.number = argument[3]
                else:
                    inst.number = argument[2]

        attack_pairs = data['attack_pairs']
        for pair in attack_pairs:
            # print(pair)
            argument1 = pair[0].split('.')
            arg1_class = argument_class_mapping[pair[0]]
            number_idx = 3 if len(argument1) == 4 else 2
            argument1 = onto.search_one(is_a=onto[arg1_class],
                                        round=argument1[1],
                                        number=argument1[number_idx])
            argument2 = pair[1].split('.')
            arg2_class = argument_class_mapping[pair[1]]
            argument2 = onto.search_one(is_a=onto[arg2_class],
                                        round=argument2[1],
                                        number=argument2[number_idx])
            argument1.attacks.append(argument2)
            argument2.isAttackedBy.append(argument1)

        # Starting reasoning to derive inverse attacks
        # sync_reasoner_pellet(infer_property_values = True,
        #                     debug = 2)

        # Closing world
        for inst in onto.individuals():
            if inst.attacks:
                inst.is_a.append(attacks.only(owl.OneOf(inst.attacks)))
            else:
                inst.is_a.append(attacks.only(owl.Nothing))
            if inst.isAttackedBy:
                inst.is_a.append(isAttackedBy.only(owl.OneOf(inst.isAttackedBy)))
            else:
                inst.is_a.append(isAttackedBy.only(owl.Nothing))

        # Defining conflict free sets
        for argument_set1 in argument_sets.keys():
            Cl = onto[argument_set1]
            Cl_cf = types.new_class(f'{argument_set1}ConflictFree', (Cl,))
            complement = []
            for argument_set2 in argument_sets.keys():
                if argument_set1 != argument_set2:
                    complement.append(onto[argument_set2])
            Cl_cf.equivalent_to.append(Cl & attacks.only(owl.Or(complement)))

        # Defining admissible sets
        for argument_set1 in argument_sets.keys():
            Cl_cf = onto[f'{argument_set1}ConflictFree']
            Cl_adm = types.new_class(f'{argument_set1}Admissible', (Cl_cf,))
            Cl_adm.equivalent_to.append(Cl_cf & isAttackedBy.only(isAttackedBy.some(Cl_cf)))

    with onto:
        owl.sync_reasoner_pellet(infer_property_values=True,
                                 debug=0)

    closed_args = []
    with onto:
        for inst in filter(lambda x: str(type(x)).endswith('Admissible'), onto.individuals()):
            closed_args.append(inst.label[0])
    return closed_args
