#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace std;

int t;
int n;
long long x, y;
double res;

int determinant(long long x1, long long y1, long long x2, long long y2, long long x3, long long y3)
{
    long long det = (x1 * y2) + (x2 * y3) + (x3 * y1) - (x1 * y3) - (x2 * y1) - (x3 * y2);
    if (det > 0)
    {
        return 1;
    }
    else if (det < 0)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

long long distance_squared(long long x1, long long y1, long long x2, long long y2)
{
    return (long long)(x1 - x2) * (x1 - x2) + (long long)(y1 - y2) * (y1 - y2);
}

struct point
{
    int no;
    long long x, y;
    double angle_radians;
};

vector<point> positions;
vector<point> hull;

double Calculate_Radians(point start_A, point end_A, point start_B, point end_B)
{
    double A_x = end_A.x - start_A.x;
    double A_y = end_A.y - start_A.y;
    double B_x = end_B.x - start_B.x;
    double B_y = end_B.y - start_B.y;
    double dot_product = A_x * B_x + A_y * B_y;
    double magnitude_A = sqrt(A_x * A_x + A_y * A_y);
    double magnitude_B = sqrt(B_x * B_x + B_y * B_y);
    if (magnitude_A == 0 || magnitude_B == 0)
        return -1;
    double cos_angle = dot_product / (magnitude_A * magnitude_B);
    if (cos_angle > 1)
        cos_angle = 1;
    if (cos_angle < -1)
        cos_angle = -1;
    double result = acos(cos_angle);
    int decimalsToRound = 10;
    result = round(result * pow(10.0, decimalsToRound)) / pow(10.0, decimalsToRound);
    return result;
}

point start;
bool angle_sort_help(const point &a, const point &b)
{
    if (a.angle_radians == b.angle_radians)
    {
        double magnitude_a = distance_squared(a.x, a.y, start.x, start.y);
        double magnitude_b = distance_squared(b.x, b.y, start.x, start.y);
        if (magnitude_a == magnitude_b)
        {
            return a.no < b.no;
        }
        else
        {
            return magnitude_a < magnitude_b;
        }
    }
    else
    {
        return a.angle_radians < b.angle_radians;
    }
}

vector<point> convex_hull_graham(int number_of_points, vector<point> &positions)
{
    int xmin = 10000007;
    int ymin = 10000007;
    int infinity = 20000014;
    int starting_index;
    point reference_point;
    for (int i = 0; i < number_of_points; i++)
    {
        if (positions[i].y < ymin)
        {
            starting_index = i;
            xmin = positions[i].x;
            ymin = positions[i].y;
        }
        else if (positions[i].y == ymin)
        {
            if (positions[i].x < xmin)
            {
                starting_index = i;
                xmin = positions[i].x;
                ymin = positions[i].y;
            }
        }
    }
    reference_point.x = xmin + infinity;
    reference_point.y = ymin;
    for (int i = 0; i < number_of_points; i++)
    {
        positions[i].angle_radians = Calculate_Radians(positions[starting_index], reference_point, positions[starting_index], positions[i]);
    }
    start = positions[starting_index];
    sort(positions.begin(), positions.end(), angle_sort_help);
    vector<point> hull;
    hull.push_back(positions[0]);
    int index = 1;
    while (index < positions.size())
    {
        if (positions[index].x != positions[0].x || positions[index].y != positions[0].y)
        {
            hull.push_back(positions[index]);
            index++;
            break;
        }
        else
        {
            index++;
        }
    }
    for (int i = index; i < positions.size(); i++)
    {
        if (positions[i].x == hull[hull.size() - 1].x && positions[i].y == hull[hull.size() - 1].y)
        {
            continue;
        }
        int turn = determinant(hull[hull.size() - 2].x, hull[hull.size() - 2].y, hull[hull.size() - 1].x, hull[hull.size() - 1].y, positions[i].x, positions[i].y);
        if (turn == 1)
        {
            hull.push_back(positions[i]);
        }
        else if (turn == 0)
        {
            hull.pop_back();
            hull.push_back(positions[i]);
        }
        else
        {
            bool still_clockwise = true;
            while (still_clockwise)
            {
                hull.pop_back();
                turn = determinant(hull[hull.size() - 2].x, hull[hull.size() - 2].y, hull[hull.size() - 1].x, hull[hull.size() - 1].y, positions[i].x, positions[i].y);
                if (turn == 1)
                {
                    still_clockwise = false;
                    hull.push_back(positions[i]);
                }
            }
        }
    }
    hull.push_back(hull[0]);
    return hull;
}

vector<point> convex_hull_chan(int number_of_points, vector<point> positions)
{
    int xmin = 10000007;
    int ymin = 10000007;
    int infinity = 20000014;
    int starting_index;
    point previous_point;
    point current_point;
    for (int i = 0; i < number_of_points; i++)
    {
        if (positions[i].y < ymin)
        {
            starting_index = i;
            xmin = positions[i].x;
            ymin = positions[i].y;
        }
        else if (positions[i].y == ymin)
        {
            if (positions[i].x < xmin)
            {
                starting_index = i;
                xmin = positions[i].x;
                ymin = positions[i].y;
            }
        }
    }
    long long subset_size = 4;
    vector<vector<point>> smaller_hulls;
    vector<point> subset;
    bool found_hull = false;
    vector<point> hull;
    while (found_hull == false)
    {
        smaller_hulls.clear();
        subset.clear();
        hull.clear();
        for (int i = 0; i < number_of_points; i++)
        {
            subset.push_back(positions[i]);
            if ((i + 1) % subset_size == 0)
            {
                smaller_hulls.push_back(convex_hull_graham(subset_size, subset));
                subset.clear();
            }
        }
        if (subset.size() != 0)
        {
            smaller_hulls.push_back(convex_hull_graham(subset.size(), subset));
            subset.clear();
        }
        hull.push_back(positions[starting_index]);
        previous_point.x = xmin - infinity;
        previous_point.y = ymin;
        current_point.x = xmin;
        current_point.y = ymin;
        int points_visited = 0;
        while (points_visited < subset_size)
        {
            point next_point;
            double minimum_angle = 10;
            double angle;
            for (int i = 0; i < smaller_hulls.size(); i++)
            {
                int LeftIndex = 0;
                int RightIndex = smaller_hulls[i].size() - 1;
                int MidPoint;
                int PointerToGreater = -1;
                int RightTangent;
                double BreakPointAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][0]);
                double BeforeBreakPointAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][smaller_hulls[i].size() - 2]);
                double AfterBreakPointAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][1]);
                if (BreakPointAngle == -1)
                {
                    LeftIndex = 0;
                    RightIndex = 1;
                }
                else if (AfterBreakPointAngle < BreakPointAngle)
                {
                    PointerToGreater = -1;
                }
                else if (AfterBreakPointAngle > BreakPointAngle)
                {
                    PointerToGreater = 1;
                }
                else
                {
                    if (BeforeBreakPointAngle < BreakPointAngle)
                    {
                        PointerToGreater = -1;
                    }
                    else if (BeforeBreakPointAngle > BreakPointAngle)
                    {
                        LeftIndex = 0;
                        RightIndex = 1;
                    }
                }
                double PreviousAngle;
                double CurrentAngle;
                while (RightIndex - LeftIndex > 1)
                {
                    MidPoint = (LeftIndex + RightIndex) / 2;
                    PreviousAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][MidPoint + PointerToGreater]);
                    CurrentAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][MidPoint]);
                    if (CurrentAngle >= BreakPointAngle)
                    {
                        if (PointerToGreater == -1)
                        {
                            RightIndex = MidPoint + PointerToGreater;
                        }
                        else if (PointerToGreater == 1)
                        {
                            LeftIndex = MidPoint + PointerToGreater;
                        }
                    }
                    else
                    {
                        if (PreviousAngle <= CurrentAngle)
                        {
                            if (PointerToGreater == -1)
                            {
                                RightIndex = MidPoint + PointerToGreater;
                            }
                            else if (PointerToGreater == 1)
                            {
                                LeftIndex = MidPoint + PointerToGreater;
                            }
                        }
                        else
                        {
                            if (PointerToGreater == -1)
                            {
                                LeftIndex = MidPoint;
                            }
                            else if (PointerToGreater == 1)
                            {
                                RightIndex = MidPoint;
                            }
                        }
                    }
                }
                PreviousAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][LeftIndex]);
                CurrentAngle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][RightIndex]);
                if (PreviousAngle < CurrentAngle)
                {
                    if (PreviousAngle == -1)
                    {
                        RightTangent = RightIndex;
                    }
                    else
                    {
                        RightTangent = LeftIndex;
                    }
                }
                else
                {
                    RightTangent = RightIndex;
                    if (CurrentAngle == -1)
                    {
                        RightTangent = RightIndex + 1;
                    }
                }
                if (RightTangent > smaller_hulls[i].size() - 1)
                {
                    RightTangent = 1;
                }
                angle = Calculate_Radians(previous_point, current_point, current_point, smaller_hulls[i][RightTangent]);
                if (angle != -1)
                {
                    if (angle < minimum_angle)
                    {
                        next_point = smaller_hulls[i][RightTangent];
                        minimum_angle = angle;
                    }
                }
            }
            points_visited++;
            if (minimum_angle < 0.01)
            {
                if (determinant(previous_point.x, previous_point.y, current_point.x, current_point.y, next_point.x, next_point.y) == 0)
                {
                    if (hull.size() != 1)
                    {
                        hull.pop_back();
                    }
                }
            }
            else if (minimum_angle == 10)
            {
                if (points_visited == subset_size)
                {
                    hull.push_back(hull[0]);
                    found_hull = true;
                    break;
                }
                else
                {
                    continue;
                }
            }
            hull.push_back(next_point);
            if (hull[hull.size() - 1].x == positions[starting_index].x && hull[hull.size() - 1].y == positions[starting_index].y)
            {
                found_hull = true;
                hull.pop_back();
                hull.push_back(hull[0]);
            }
            previous_point = current_point;
            current_point = hull[hull.size() - 1];
            if (found_hull == true)
            {
                break;
            }
        }
        subset_size = subset_size * subset_size;
    }
    return hull;
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> t;
    while (t--)
    {
        positions.clear();
        hull.clear();
        res = 0;
        cin >> n;
        for (int i = 0; i < n; i++)
        {
            cin >> x >> y;
            point new_sheep;
            new_sheep.no = i + 1;
            new_sheep.x = x;
            new_sheep.y = y;
            positions.push_back(new_sheep);
        }
        hull = convex_hull_chan(n, positions);
        if (positions.size() > 1)
            res += sqrt((double)(hull[hull.size() - 1].x - hull[0].x) * (hull[hull.size() - 1].x - hull[0].x) + (hull[hull.size() - 1].y - hull[0].y) * (hull[hull.size() - 1].y - hull[0].y));
        for (int j = 1; j < (int)hull.size(); j++)
        {
            res += sqrt((double)(hull[j].x - hull[j - 1].x) * (hull[j].x - hull[j - 1].x) + (hull[j].y - hull[j - 1].y) * (hull[j].y - hull[j - 1].y));
        }
        cout << fixed << setprecision(2) << res << endl;
        for (int j = 0; j < hull.size() - 1; j++)
            cout << hull[j].no << " ";
        cout << endl;
        cout << endl;
    }
}
