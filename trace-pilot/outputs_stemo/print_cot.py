import json


def main():
    with open("for_inspect.json", 'r') as f:
        output_dict = json.load(f)
    
    print(output_dict['video_id'])
    print('-' * 50)
    print(output_dict['thinking_trace'])



if __name__ == '__main__':
    main()