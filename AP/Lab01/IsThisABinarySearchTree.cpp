/* Hidden stub code will pass a root argument to the function below. Complete the function to solve the challenge. Hint: you may want to write one or more helper functions.

The Node struct is defined as follows:
    struct Node {
        int data;
        Node* left;
        Node* right;
    }
*/

bool helper(Node *root, int mn, int mx)
{
    if (root == NULL)
        return true;
    if (root->data < mn || root->data > mx)
        return false;
    return (helper(root->left, mn, root->data - 1) && helper(root->right, root->data + 1, mx));
}
bool checkBST(Node *root)
{
    return helper(root, 0, 10000);
}