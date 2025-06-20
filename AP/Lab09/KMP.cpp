#include <iostream>
#include <vector>

using namespace std;

int t;
int p;
vector<char> pattern;
vector<char> text;
char c;
string help;
vector<int> PrefixSuffixArray;

vector<int> PrefixSuffix(vector<char> pattern)
{
    vector<int> pi;
    int length = pattern.size();
    pi.push_back(0);
    int PrefixIndex = 0;
    bool updated;
    for (int i = 1; i < length; i++)
    {
        if (pattern[i] == pattern[PrefixIndex])
        {
            pi.push_back(PrefixIndex + 1);
            PrefixIndex++;
        }
        else
        {
            updated = false;
            while (PrefixIndex != 0)
            {
                PrefixIndex = pi[PrefixIndex - 1];
                if (pattern[i] == pattern[PrefixIndex])
                {
                    pi.push_back(PrefixIndex + 1);
                    PrefixIndex++;
                    updated = true;
                    break;
                }
            }
            if (updated == false)
            {
                pi.push_back(0);
            }
        }
    }
    return pi;
}

vector<int> KnuthMorrisPratt(vector<char> text, vector<char> pattern)
{
    vector<int> occurrences;
    vector<int> PrefixSuffixArray = PrefixSuffix(pattern);
    bool updated;
    int TextIndex = 0;
    int PatternIndex = 0;
    int PatternLength = pattern.size();
    while (TextIndex < text.size())
    {
        if (pattern[PatternIndex] == text[TextIndex])
        {
            TextIndex++;
            PatternIndex++;
        }
        else
        {
            updated = false;
            while (PatternIndex != 0)
            {
                PatternIndex = PrefixSuffixArray[PatternIndex - 1];
                if (pattern[PatternIndex] == text[TextIndex])
                {
                    TextIndex++;
                    PatternIndex++;
                    updated = true;
                    break;
                }
            }
            if (updated == false)
            {
                TextIndex++;
            }
        }
        if (PatternIndex == PatternLength)
        {
            occurrences.push_back(TextIndex - PatternLength);
            PatternIndex = PrefixSuffixArray[PatternIndex - 1];
        }
    }
    return occurrences;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        cin >> p;
        pattern.clear();
        text.clear();
        PrefixSuffixArray.clear();
        for (int i = 0; i < p; i++)
        {
            cin >> c;
            pattern.push_back(c);
        }
        PrefixSuffixArray = PrefixSuffix(pattern);
        cin >> help;
        for (int i = 0; i < (int)help.size(); i++)
        {
            text.push_back(help[i]);
        }
        vector<int> positions = KnuthMorrisPratt(text, pattern);
        for (int i = 0; i < positions.size(); i++)
        {
            cout << positions[i] << endl;
        }
    }
}
