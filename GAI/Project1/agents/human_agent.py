from typing import Dict, Tuple
import torch


class HumanAgent:
    def __init__(self):
        self.name = "human"

    def select_action(self, next_states: Dict[Tuple[int, int], torch.FloatTensor]) -> Tuple[int, int]:
        """
        next_states:
            dict gdzie klucz = (x, rotation)
            wartość = tensor cech stanu po ruchu
        """
        actions = sorted(next_states.keys())

        print("\nLegal moves:")
        for idx, action in enumerate(actions):
            x, rot = action
            state = next_states[action].tolist()
            print(
                f"[{idx}] x={x}, rot={rot}, "
                f"lines={state[0]:.0f}, holes={state[1]:.0f}, "
                f"bumpiness={state[2]:.0f}, height={state[3]:.0f}"
            )

        while True:
            user_input = input("Enter move number: ").strip()
            try:
                choice = int(user_input)
                if 0 <= choice < len(actions):
                    return actions[choice]
                else:
                    print("Illegal move number")
            except ValueError:
                print("Illegal ASCII character")