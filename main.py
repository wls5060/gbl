import argparse
import propagation
def main(args) :
    print(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--root", type=str, default="./")
    parser.add_argument("--gpu", type=int, default=3)
    
    args = parser.parse_args()

    main(args)