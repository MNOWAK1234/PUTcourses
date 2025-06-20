#include <iostream>
#define letters 26
#define PS prefixsearch

using namespace std;

int n, k;
string word;
bool help;
bool whatever;
struct trienode
{
    char value;
    trienode *children[letters];
    bool stop;
};
trienode *root;
trienode *getnode(int index)
{
    trienode *newnode = new trienode;
    newnode->value = char(int('a') + index);
    newnode->stop = false;
    for (int i = 0; i < letters; i++)
        newnode->children[i] = NULL;
    return newnode;
}
void init()
{
    root = getnode(char(int('/') - int('a')));
}
void add(string word)
{
    trienode *curr = root;
    int index;
    for (int i = 0; i < word.size(); i++)
    {
        index = int(word[i]) - int('a');
        if (curr->children[index] == NULL)
            curr->children[index] = getnode(index);
        curr = curr->children[index];
    }
    curr->stop = true;
}
void output(trienode *curr, string word)
{
    if (curr->stop == true)
    {
        cout << word << "\n";
        whatever = true;
    }
    for (int i = 0; i < letters; i++)
    {
        if (curr->children[i] != NULL)
        {
            output(curr->children[i], word + curr->children[i]->value);
        }
    }
}
void prefixsearch(string word)
{
    trienode *curr = root;
    int index;
    for (int i = 0; i < word.size(); i++)
    {
        index = int(word[i]) - int('a');
        if (curr->children[index] == NULL)
        {
            cout << "No match." << "\n";
            return;
        }
        curr = curr->children[index];
    }
    help = curr->stop;
    curr->stop = false;
    whatever = false;
    output(curr, word);
    if (whatever == false)
        cout << "No match." << "\n";
    curr->stop = help;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin >> n;
    init();
    while (n--)
    {
        cin >> word;
        add(word);
    }
    cin >> k;
    for (int i = 1; i <= k; i++)
    {
        cout << "Case #" << i << ":\n";
        cin >> word;
        PS(word);
    }
}
