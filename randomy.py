import random

def randomize_array(arr):

    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

def main():

    presenter = ["井上", "金井", "木村", "土屋", "肌附", "松上"]
    print("Members: ", presenter)

    randomize_array(presenter)
    print("Randomized turn:", presenter)

if __name__ == "__main__":
    main()