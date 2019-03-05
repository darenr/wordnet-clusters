import json
import sys

if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            print sys.argv[1]
            j = json.loads(f.read())
            for cluster in j:
                if len(j[cluster]) < 20 and len(j[cluster]) > 3:
                    print '-' * 75
                    print(', '.join(sorted(j[cluster])))

    else:
        print "usage: <json file>"
