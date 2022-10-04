#include <igl/readOFF.h>
#include <igl/writeSTL.h>
//#undef IGL_STATIC_LIBRARY
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <math.h> 
using namespace Eigen;
using namespace std;


Eigen::MatrixXd VA,VB,VC;
Eigen::VectorXi J,I;
Eigen::MatrixXd N1, N2;
Eigen::MatrixXi FA,FB,FC;
igl::MeshBooleanType boolean_type(
  igl::MESH_BOOLEAN_TYPE_UNION);

const char * MESH_BOOLEAN_TYPE_NAMES[] =
{
  "Union",
  "Intersect",
  "Minus",
  "XOR",
  "Resolve",
};

void update(igl::opengl::glfw::Viewer &viewer)
{
  igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC,J);
  Eigen::MatrixXd C(FC.rows(),3);
  for(size_t f = 0;f<C.rows();f++)
  {
    if(J(f)<FA.rows())
    {
      C.row(f) = Eigen::RowVector3d(1,0,0);
    }else
    {
      C.row(f) = Eigen::RowVector3d(0,1,0);
    }
  }
  viewer.data().clear();
  viewer.data().set_mesh(VC,FC);
  viewer.data().set_colors(C);
  std::cout<<"A "<<MESH_BOOLEAN_TYPE_NAMES[boolean_type]<<" B."<<std::endl;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    default:
      return false;
    case '.':
      boolean_type =
        static_cast<igl::MeshBooleanType>(
          (boolean_type+1)% igl::NUM_MESH_BOOLEAN_TYPES);
      break;
    case ',':
      boolean_type =
        static_cast<igl::MeshBooleanType>(
          (boolean_type+igl::NUM_MESH_BOOLEAN_TYPES-1)%
          igl::NUM_MESH_BOOLEAN_TYPES);
      break;
    case '[':
      viewer.core().camera_dnear -= 0.1;
      return true;
    case ']':
      viewer.core().camera_dnear += 0.1;
      return true;
  }
  update(viewer);
  return true;
}

void connect_2_meshes(std::string m1, std::string m2, std::string m) {
    Eigen::MatrixXd v1, v2;
    Eigen::MatrixXi f1, f2;

    igl::read_triangle_mesh(m1, v1, f1);
    igl::read_triangle_mesh(m2, v2, f2);

    MatrixXd V(v1.rows() + v2.rows(), v1.cols());
    V << v1, v2;

    int v1_rows = v1.rows();
    for (int i = 0; i < f2.rows(); i++) {
        for (int j = 0; j < 3; j++)
            f2(i, j) += v1_rows;
    }
    MatrixXi F(f1.rows() + f2.rows(), f1.cols());
    F << f1, f2;

    igl::writeSTL(m, V, F);

    //pausee();
}

std::tuple<MatrixXd, MatrixXi> connect_meshes(Eigen::MatrixXd v1, Eigen::MatrixXd v2, Eigen::MatrixXi f1, Eigen::MatrixXi f2) {

    MatrixXd V(v1.rows() + v2.rows(), v1.cols());
    V << v1, v2;

    int v1_rows = v1.rows();
    int v2_rows = v2.rows();
    for (int i = 0; i < f2.rows(); i++) {
        for (int j = 0; j < 3; j++)
            f2(i, j) += v1_rows;
    }

    MatrixXi F(f1.rows() + f2.rows(), f1.cols());
    F << f1, f2;

    return { V, F };
}

Eigen::MatrixXd move_mesh(Eigen::MatrixXd v1, double x, double y, double z) {

    for (int i = 0; i < v1.rows(); i++) {
        v1(i, 0) += x;
        v1(i, 1) += y;
        v1(i, 2) += z;
    }

    return v1;
}

// A program to scale the sphere from radius 40mm to the desired size
// new size = 40.0 * scale
void scale_mesh(Eigen::MatrixXd& v1, double scale) {
    for (int i = 0; i < v1.rows(); i++) {
        v1(i, 0) *= scale;
        v1(i, 1) *= scale;
        v1(i, 2) *= scale;
    }
}



vector<double> linspace(double a, double b, int num) {
    vector<double> vect(num);
    for (int i = 0; i < num; ++i) {
        double tmp = (double)((a + (b - a) / (double)num / 2.0) + (double)(b - a) / num * i);
        vect[i] = tmp;
    }
    return vect;
}

vector<double> circlespace(double radius, int num) {
    vector<double> vect(num);
    for (int i = 0; i < num; ++i) {
        double tmp = radius * cos(2 * (3.1416) / num * i);
        vect[i] = tmp;
    }
    return vect;
}

void generate_meshB() {
    Eigen::MatrixXd v1, v2;
    Eigen::MatrixXi f1, f2;

    /* Generate material B STL files at specified location*/
    double BALL_SIZE = 0;
    double A_SIZE = 200;
    double GAP_DISTANCE = 0;
    double Volume_A = pow(A_SIZE, 3);
    std::cout << "Please enter the radius of material B spheres and press ENTER: ";
    cin >> BALL_SIZE;
    std::cout << "Please enter the gap distance between material B spheres and press ENTER: ";
    cin >> GAP_DISTANCE;

    while (GAP_DISTANCE < 10) {
        std::cout << "Please enter a gap distance larger than 10: ";
        cin >> GAP_DISTANCE;
    }

    int B_Number = Volume_A / ((BALL_SIZE * 2 + GAP_DISTANCE) * (BALL_SIZE * 2 + GAP_DISTANCE) * (BALL_SIZE * 2 + GAP_DISTANCE));
    std::cout << B_Number << "maximum of material B spheres can be generated inside material A" << endl;
    std::cout << "Generating......" << endl;

    B_Number = 32; // set it to 32 for demo;
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_ball.stl", v1, f1);
    scale_mesh(v1, BALL_SIZE / 40.0); // v1 will be modified

    // Create material B meshes now
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;
    // Linear material B pattern
    vector<double> space = linspace(-A_SIZE * 1.5, A_SIZE * 1.5, 6);

    //Eigen::MatrixXd v_tmp1 = move_mesh(v1, 50, 50, 50);
    //Eigen::MatrixXd v_tmp2 = move_mesh(v1, -50, -50, -50);
    //std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, space[i], space[j], A_SIZE * 0.5);
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, space[j], space[i], A_SIZE * 0.5);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, space[i], space[j], A_SIZE * 0.3);
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, space[j], space[i], A_SIZE * 0.3);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, space[i], space[j], A_SIZE * -0.15);
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, space[j], space[i], A_SIZE * -0.15);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }

    // // Ring shape pattern
    //int num = 10, radius = 100;
    //for (int i = 0; i < num; ++i) {
    //    Eigen::MatrixXd v_tmp1 = move_mesh(v1, radius * cos(2 * (3.1416) / num * i), radius * sin(2 * (3.1416) / num * i), 0);
    //    Eigen::MatrixXd v_tmp2 = move_mesh(v1, radius * cos(2 * (3.1416) / num * i), radius * sin(2 * (3.1416) / num * i), A_SIZE * -0.3);
    //    std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
    //    meshB.push_back(mesh_matrix);
    //}

    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;

    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }

    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB_large_cube.stl", get<0>(tmp1), get<1>(tmp1));
}

void combine_gcode() {
    char oldname[] = "C:/Users/Derek Zhang/Desktop/merge_model.gcode";
    char newname[] = "C:/Users/Derek Zhang/Desktop/merge_model.txt";
    string modelAname = ";MESH:Cube-ball3.stl";
    string modelBname = ";MESH:ball-assembly.STL";

    /*	Deletes the file if exists */
    if (rename(oldname, newname) != 0)
        std::cout << "file already exists" << endl;
    else
        std::cout << "Change material A gcode file to .txt successfully" << endl;


    string filename = "C:/Users/Derek Zhang/Desktop/merge_model.txt";
    // open input file
    std::ifstream readFile(filename);
    // create output file
    std::ofstream outfile("C:/Users/Derek Zhang/Desktop/DualModel.txt");

    if (!readFile.is_open()) std::cout << "error opening!" << endl;
    else std::cout << "reading file..." << endl;

    // Start to process the files
    string str = "";
    int material_status = 0; // status 0 for material A, status 1 for material B

    while (str != modelAname) {
        getline(readFile, str); // find the begining of the toolpath
    }
    outfile << str << endl;

    while (getline(readFile, str)) {

        if (str.find(";LAYER") != string::npos) continue; // skip layer information
        else if (str == modelAname) {
            if (material_status == 1) { // switch from material B to material A
                material_status = 0;
                outfile << ";switch material A" << endl;
                outfile << "G04 P15000" << endl;
            }
            else if (material_status == 0) outfile << "G04 P10000" << endl;
        }
        else if (str == modelBname) {
            if (material_status == 0) { // switch from material A to material B
                material_status = 1;
                outfile << ";switch material B" << endl;
            }
            outfile << "G04 P15000" << endl;
        }

        outfile << str << endl;

    }

    outfile.close();

}

int main(int argc, char *argv[])
{
    //test git push
    Eigen::MatrixXd v1, v2, v3;
    Eigen::MatrixXi f1, f2, f3;

    igl::readOFF("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Shell/Cube-shell.off",v1,f1);
    igl::readOFF("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Shell/meshB_large_circle.off",v2,f2);

    igl::copyleft::cgal::mesh_boolean(v1, f1, v2, f2, igl::MESH_BOOLEAN_TYPE_MINUS, v3, f3);
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Shell/testBOOLEAN.STL", v3, f3);

    //igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Cube.STL", v1, f1);
    //igl::writeOFF("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Cube-test.off", v1, f1);


    /*igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(v3, f3);
    viewer.data().set_face_based(true);
    viewer.launch();*/

    std::cout << "Finished !" << endl;
    return EXIT_SUCCESS;
}
