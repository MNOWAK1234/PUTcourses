#include <iostream>
#include <vector>
#include <stack>

using namespace std;

int n;
int num;
stack<int> killer;
stack<int> value;
int mx;
int curr;
int help;

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    killer.push(2000000000);
    value.push(-1);
    while (n--)
    {
        cin >> num;
        if (num > killer.top() && value.top() != 1)
        {
            killer.push(num);
            value.push(1);
        }
        else if (num > killer.top())
        {
            killer.pop();
            killer.push(num);
        }
        else
        {
            value.push(9);
            while (!killer.empty() && killer.top() >= num)
            {
                killer.pop();
                value.pop();
            }
            curr = value.top();
            value.pop();
            if (killer.empty())
            {
                value.push(0);
            }
            else
            {
                if (curr + 1 != value.top())
                    value.push(curr + 1);
                else
                    killer.pop();
            }
            killer.push(num);
        }
        if (value.top() > mx)
            mx = value.top();
    }
    cout << mx << endl;
}
