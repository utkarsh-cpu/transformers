


def edit_two_letters(word, allow_switches = True):
    '''
    Input:
        word: the input string/word 
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''
    
    edit_two_set = set()
    
    ### START CODE HERE ###
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_l =[(word[:i],word[i:]) for i in range(len(word)+1)]
    if allow_switches:
        edit_one_set=set([L+i+R for L,R in split_l for i in letters]
        +sorted([L.replace(L[-1],i)+R for L,R in split_l if L for i in letters if i!=L[-1]])
        +[L+R[1]+R[0]+R[2:] for L,R in split_l if len(R)>1]
        +[L+R[1:] for L,R in split_l if R])
        for l in edit_one_set:
            split_u =[(l[:i],l[i:]) for i in range(len(l)+1)]
            edit_two_set.update([L+i+R for L,R in split_u for i in letters]+sorted([L.replace(L[-1],i)+R for L,R in split_u if L for i in letters if i!=L[-1]])+[L+R[1]+R[0]+R[2:] for L,R in split_u if len(R)>1]+[L+R[1:] for L,R in split_u if L and R])
    else:
        edit_one_set=set([L+i+R for L,R in split_l for i in letters]
        +sorted([L.replace(L[-1],i)+R for L,R in split_l if L for i in letters if i!=L[-1]])
        +[L+R[1:] for L,R in split_l if R])
        for l in edit_one_set:
            split_u =[(l[:i],l[i:]) for i in range(len(l)+1)]
            edit_two_set.update([L+i+R for L,R in split_u for i in letters]+sorted([L.replace(L[-1],i)+R for L,R in split_u if L for i in letters if i!=L[-1]])+[L+R[1:] for L,R in split_u if L and R])
    if len(word)<=2:
        edit_two_set.add('')
    ### END CODE HERE ###
    
    # return this as a set instead of a list
    return edit_two_set

print(len(edit_two_letters('cat',allow_switches=False)))