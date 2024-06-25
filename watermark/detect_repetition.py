import re

def longest_repeated_substring(str):
    n = len(str)
    LCSRe = [[0 for x in range(n + 1)]
                for y in range(n + 1)]
 
    res = "" # To store result
    res_length = 0 # To store length of result
 
    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
             
            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (str[i - 1] == str[j - 1] and
                LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1
 
                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)
                 
            else:
                LCSRe[i][j] = 0
 
    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                                    index + 1):
            res = res + str[i - 1]
 
    return res

def remove_min_repeated_substring(s):
    repeats = re.findall(r'(.+?)\1+', s)
    
    # Sort the repeats by length, in descending order
    repeats = sorted(repeats, key=len, reverse=True)
    
    for repeat in repeats:
        while True:
            # Replace only once in each iteration
            new_s = s.replace(repeat * 2, repeat, 1)

            # If the string didn't change, we're done with this repeat
            if new_s == s:
                break
            else:
                s = new_s

    return s

def shortest_repeated_substring(s):
    # Find all repeated substrings
    repeats = re.findall(r'(.+?)\1+', s)

    if repeats:
        # Sort the repeats by length, in ascending order
        repeats = sorted(repeats, key=len)

        # The shortest repeated substring is the first element
        shortest_repeat = repeats[0]

        # Replace all instances of the repeated substring with a single instance
        s = s.replace(shortest_repeat * 2, shortest_repeat)

    return s

def remove_repeated_substrings(s):
    while True:
        # Find and remove the shortest repeated substring
        new_s = shortest_repeated_substring(s)

        # If the string didn't change, we're done
        if new_s == s:
            break
        else:
            s = new_s

    return s

# Test the function
str = "1.666666666666666"
# print(longest_repeated_substring(str))  # Output: 6666666666666
# print(remove_min_repeated_substring(str)) 
print(remove_repeated_substrings(str)) 

str = "If it has 25 in the first place and if it has 25 - 1 in the end then it has 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,"
# print (longest_repeated_substring(str))
# print(remove_min_repeated_substring(str))
print(remove_repeated_substrings(str))

str = "If in the race with 240 people there is 80 who is Japanese and rest who is Chinese then that means of 80 who is Japanese 60 will be boy and rest will be girl and if rest who is Chinese then that means of 160 who is rest who is Chinese then that means of 160 who is rest who is Chinese then that means of 160 who is rest who is Chinese then that means of 160 who is rest who is Chinese then that means of 160 who is rest who is Chinese then that means of 16"
# print (longest_repeated_substring(str))
# print(remove_min_repeated_substring(str))
print(remove_repeated_substrings(str))

