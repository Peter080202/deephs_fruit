import os
from typing import List
from core.name_convention import *


# CLASS_TO_FRUIT = {
#     "class_0": AppleType.GRANNY_SMITH,
#     "class_1": AppleType.GRANNY_SMITH,
#     "class_2": AppleType.GRANNY_SMITH,
#     "class_3": AppleType.GRANNY_SMITH,
#     "class_4": AppleType.GRANNY_SMITH,
#     "class_5": AppleType.JOYA,
#     "class_6": AppleType.ENVY,
#     "class_7": AppleType.GRANNY_SMITH,
#     "class_8": AppleType.JOYA,
#     "class_9": AppleType.GOLDEN_DELICIOUS
# }


def detect_side(filename: str) -> Side:
    if filename.endswith("000.tif"):
        return Side.FRONT
    if filename.endswith("090.tif"):
        return Side.SIDE
    if filename.endswith("180.tif"):
        return Side.BACK
    if filename.endswith("270.tif"):
        return Side.SIDE
    raise ValueError(f"Unknown side: {filename}")

def generate_apple_records(dataset_root: str) -> List[FruitRecord]:
    records = []

    print(dataset_root)

    for class_folder in os.listdir(dataset_root):
        full_path = os.path.join(dataset_root, class_folder)

        print(class_folder)
        if not os.path.isdir(full_path):
            continue

        # fruit_type = CLASS_TO_FRUIT[class_folder]

        for file in os.listdir(full_path):
            if not file.endswith(".tif"):
                print("not valid")
                print(file)
                continue

            print("valid")
            print(file)
            # extract info
            side = detect_side(file)

            print(side)

            r = FruitRecord(
                fruit=Fruit.APPLE,
                side=side,
                day=Day.DAY_1,
                id=ID.UNKNOWN,
                camera_type=CameraType.VIS,
                classtype=class_folder,
                label= AppleLabel("Not infected" if class_folder == 'class_0' else "Infected"),
                filename=os.path.join(full_path, file)
            )

            records.append(r)

    return records
