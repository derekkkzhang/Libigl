#include <igl/readOFF.h>
#include <igl/writeSTL.h>
#include <igl/cylinder.h>
#include <igl/upsample.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/copyleft/cgal/points_inside_component.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h> 
#include <vector>
#include <cstdio>
#include <queue>

// For CGAL remesh
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Mesh_3/dihedral_angle_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <boost/iterator/function_output_iterator.hpp>

// CGAL Remeshing declaration
typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef CGAL::Surface_mesh<K::Point_3>                        Mesh;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;

struct halfedge2edge
{
    halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges)
        : m_mesh(m), m_edges(edges)
    {}
    void operator()(const halfedge_descriptor& h) const
    {
        m_edges.push_back(edge(h, m_mesh));
    }
    const Mesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};

// For CGAL ovoid mesh
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>

// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;
typedef FT(*Function)(Point_3);
typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;


// For uniform points distribution
#include <CGAL/point_generators_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include "utils_sampling.hpp"
#include "vcg_mesh.hpp"
#include <WAAM_compute_spacing.h>
#include <igl/rotation_matrix_from_directions.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/boost/graph/iterator.h>
#include <queue>
#include <unordered_map>
#include <CGAL/IO/write_points.h>
#include <optional>
#include <CGAL/centroid.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/draw_triangulation_3.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/centroid.h>

using namespace Utils_sampling;
typedef CGAL::Polyhedron_3<K>                              Polyhedron;
typedef K::Point_3                                         Point;
typedef K::Vector_3                                        Vector_3;
typedef CGAL::Point_set_3<Point>                           Point_set;
typedef CGAL::Parallel_if_available_tag                    Concurrency_tag;
typedef boost::graph_traits<Mesh>::vertex_descriptor       vertex_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor         face_descriptor;
typedef boost::graph_traits<Mesh>::vertex_iterator         vertex_iterator;
typedef CGAL::Halfedge_around_target_iterator<Mesh>        halfedge_around_target_iterator;
typedef std::array<std::size_t, 3>                         Facet; // Triple of indices
typedef CGAL::Advancing_front_surface_reconstruction<K>    Surface_reconstruction;
typedef CGAL::Delaunay_triangulation_3<K>                  DT3;

// For k-neighbor searching
typedef CGAL::Search_traits_3<K>                                           Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>                         K_neighbor_search;
typedef K_neighbor_search::Tree                                            Tree;
typedef Tree::Splitter                                                     Splitter;
typedef K_neighbor_search::Distance                                        Distance;

const Point null_point = Point_3(0, 0, 0);

struct Edge {
    Point_3 source;
    Point_3 target;

    Edge(const Point_3& s, const Point_3& t) : source(s), target(t) {}
};

struct pair_hash { // for edge unordered_set
    inline std::size_t operator()(const std::pair<Point, Point>& p) const {
        std::size_t h1 = std::hash<Point>{}(p.first);
        std::size_t h2 = std::hash<Point>{}(p.second);
        return h1 ^ (h2 << 1); // combine hashes
    }
};


FT sphere_function(Point_3 p) {
    const FT x2 = p.x() * p.x(), y2 = p.y() * p.y(), z2 = p.z() * p.z();
    return x2 + y2 + z2 - 1;
}


// For reconstruct surface_mesh from a set of points
struct Construct {
    Mesh& mesh;
    template < typename PointIterator>
    Construct(Mesh& mesh, PointIterator b, PointIterator e)
        : mesh(mesh)
    {
        for (; b != e; ++b) {
            boost::graph_traits<Mesh>::vertex_descriptor v;
            v = add_vertex(mesh);
            mesh.point(v) = *b;
        }
    }
    Construct& operator=(const Facet f)
    {
        typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
        typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
        mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
            vertex_descriptor(static_cast<size_type>(f[1])),
            vertex_descriptor(static_cast<size_type>(f[2])));
        return *this;
    }
    Construct&
        operator*() { return *this; }
    Construct&
        operator++() { return *this; }
    Construct
        operator++(int) { return *this; }
};


using namespace Eigen;
using namespace std;
#define PI 3.1415926
double TARGET_EDGE_LENGTH = 5.0;


// Global variables
Eigen::MatrixXd main_v1, main_v2, main_v3, main_v4;
Eigen::MatrixXi main_f1, main_f2, main_f3, main_f4;

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
    if (v1 == v2) {
        return { v1, f1 };
    }
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

// For material A shifting move function
void MOVE_MESH(Eigen::MatrixXd& v1, double x, double y, double z) {

    for (int i = 0; i < v1.rows(); i++) {
        v1(i, 0) += x;
        v1(i, 1) += y;
        v1(i, 2) += z;
    }
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

Eigen::MatrixXd rotate_mesh(Eigen::MatrixXd v1, Eigen::Vector3d Nvec) {
    Eigen::Vector3d vec(0, 0, 1);
    Matrix3d RM = igl::rotation_matrix_from_directions(vec, Nvec);

    /*for (int i = 0; i < v1.rows(); i++) {
        v1(i, 0) = RM(0, 0) * v1(i, 0) + RM(0, 1) * v1(i, 1) + RM(0, 2) * v1(i, 2);
        v1(i, 1) = RM(1, 0) * v1(i, 0) + RM(1, 1) * v1(i, 1) + RM(1, 2) * v1(i, 2);
        v1(i, 2) = RM(2, 0) * v1(i, 0) + RM(2, 1) * v1(i, 1) + RM(2, 2) * v1(i, 2);
    }*/
    v1 *= RM.transpose();

    return v1;
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

void CGAL_remesh(string filename) {
    Mesh mesh;
    if (!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
        std::cerr << "Invalid input." << std::endl;
        return;
    }

    unsigned int nb_iter = 10;
    std::cout << "Enter the remesh iterations (default 10, higher indicates better quality and longer time): ";
    cin >> nb_iter;

    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
    PMP::split_long_edges(border, TARGET_EDGE_LENGTH, mesh);

    std::cout << "Start remeshing of " << filename
        << " (" << num_faces(mesh) << " faces)..." << std::endl;

    // give each vertex a name, the default is empty
    Mesh::Property_map<edge_descriptor, bool> is_constrained =
        mesh.add_property_map<edge_descriptor, bool>("e:is_constrained", false).first;

    //detect sharp features
    BOOST_FOREACH(edge_descriptor e, edges(mesh))
    {
        halfedge_descriptor hd = halfedge(e, mesh);
        if (!is_border(e, mesh)) {
            double angle = CGAL::Mesh_3::dihedral_angle(mesh.point(source(hd, mesh)),
                mesh.point(target(hd, mesh)),
                mesh.point(target(next(hd, mesh), mesh)),
                mesh.point(target(next(opposite(hd, mesh), mesh), mesh)));
            if (CGAL::abs(angle) < 100)
                is_constrained[e] = true;
        }
    }

    PMP::isotropic_remeshing(faces(mesh), TARGET_EDGE_LENGTH, mesh,
        CGAL::parameters::number_of_iterations(nb_iter)
        .edge_is_constrained_map(is_constrained)); //i.e. protect border, here

    CGAL::IO::write_polygon_mesh(filename, mesh);
}

void generate_outer_meshB() {
    Eigen::MatrixXd v1, v2, v3;
    Eigen::MatrixXi f1, f2, f3;

    /* Generate material B STL files at specified location*/
    double B_SIZE = 0;
    double A_SIZE = 0;
    double GAP_DISTANCE = 0;

    int exit_loop = 0;
    std::cout << "Select material B shape (1 for sphere,  2 for cylinder, 3 for cube): ";
    while (exit_loop == 0) {
        int B_TYPE;
        cin >> B_TYPE;
        switch (B_TYPE)
        {
        default:
            std::cout << "Wrong number! Please enter a VALID material B type: ";
            break;
        case 1:
            std::cout << "Sphere shape selected!" << std::endl;
            igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_ball.stl", v1, f1);
            exit_loop = 1;
            break;
        case 2:
            std::cout << "Cylinder shape selected!" << std::endl;
            igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_cylinder.stl", v1, f1);
            exit_loop = 1;
            break;
        case 3:
            std::cout << "Cube shape selected!" << std::endl;
            igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_cube.stl", v1, f1);
            exit_loop = 1;
            break;
        }
    }

    std::cout << "Please enter the bounding box side length press ENTER: ";
    cin >> A_SIZE;
    std::cout << "Please enter the size of material B and press ENTER (default 40): ";
    cin >> B_SIZE;
    std::cout << "Please enter the gap distance between material B and press ENTER: ";
    cin >> GAP_DISTANCE;

    int B_number = A_SIZE / (B_SIZE + GAP_DISTANCE); // number of material B at one edge
    std::cout << pow(B_number, 3) << " of material B will be generated ..." << endl;
    scale_mesh(v1, B_SIZE / 40.0); // v1 will be modified

    // Create material B meshes now
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    // Linear material B pattern
    vector<double> space = linspace(-A_SIZE / 2, A_SIZE / 2, B_number);

    Matrix<double, Dynamic, 1> d;
    Matrix<double, Dynamic, 1> inside;
    Matrix<int, Dynamic, 1> I;
    Matrix<double, Dynamic, 3> C;
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Cube.STL", v2, f2);

    for (int layer = 0; layer < B_number; ++layer) {
        for (int i = 0; i < B_number; ++i) {
            for (int j = 0; j < B_number; ++j) {
                Eigen::MatrixXd v_tmp1 = move_mesh(v1, space[i], space[j], space[layer]);
                Eigen::MatrixXd v_tmp2 = move_mesh(v1, space[j], space[i], space[layer]);
                igl::point_mesh_squared_distance(v_tmp1, v2, f2, d, I, C);
                igl::copyleft::cgal::points_inside_component(v2, f2, v_tmp1, inside);
                if (*std::min_element(d.begin(), d.end()) < 400 && inside[0] == 1) {
                    std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
                    meshB.push_back(mesh_matrix);
                }
                else {
                    continue;
                }
            }
        }
    }


    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;

    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }

    // Save material B within material A boundary
    //igl::copyleft::cgal::mesh_boolean(v2, f2, get<0>(tmp1), get<1>(tmp1), igl::MESH_BOOLEAN_TYPE_INTERSECT, v3, f3);
    //igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB__OuterShell_TEST1.stl", v3, f3);
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB__OuterShell_TEST.stl", get<0>(tmp1), get<1>(tmp1));

}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_cylinder(double diameter, double height) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    double radius = diameter * 0.5;
    int axis_divisions = 2 * PI * radius / TARGET_EDGE_LENGTH;
    int height_divisions = 2;

    char ans;
    std::cout << "Do you want to customize cylinder mesh size (Y/N) ? ";
    cin >> ans;
    if (ans == 'Y') {
        std::cout << "Please enter the radial divisions number and height division number: ";
        cin >> axis_divisions >> height_divisions;
    }
    else { std::cout << "Use default cylinder mesh generation" << endl; }
    height_divisions++; // compensate for user input

    // Cylinder geometry valid check
    int threshold_angle = 5;
    double mesh_angle = TARGET_EDGE_LENGTH / (PI * diameter) * 360;
    if (mesh_angle < threshold_angle) axis_divisions = 50;

    V.resize(axis_divisions * height_divisions + 2, 3);
    F.resize(2 * (axis_divisions * (height_divisions - 1)) + 2 * axis_divisions, 3);
    int f = 0;
    for (int th = 0; th < axis_divisions; th++) {
        double x = radius * cos(2. * PI * double(th) / double(axis_divisions));
        double y = radius * sin(2. * PI * double(th) / double(axis_divisions));
        for (int h = 0; h < height_divisions; h++)
        {
            double z = height * double(h) / double(height_divisions - 1);
            V(th + h * axis_divisions, 0) = x;
            V(th + h * axis_divisions, 1) = y;
            V(th + h * axis_divisions, 2) = z;
            if (h > 0)
            {
                F(f, 0) = ((th + 0) % axis_divisions) + (h - 1) * axis_divisions;
                F(f, 1) = ((th + 1) % axis_divisions) + (h - 1) * axis_divisions;
                F(f, 2) = ((th + 0) % axis_divisions) + (h + 0) * axis_divisions;
                f++;
                F(f, 0) = ((th + 1) % axis_divisions) + (h - 1) * axis_divisions;
                F(f, 1) = ((th + 1) % axis_divisions) + (h + 0) * axis_divisions;
                F(f, 2) = ((th + 0) % axis_divisions) + (h + 0) * axis_divisions;
                f++;
            }
        }
    }

    // Top/bottom face center point
    V(axis_divisions * height_divisions, 0) = 0;
    V(axis_divisions * height_divisions, 1) = 0;
    V(axis_divisions * height_divisions, 2) = 0;
    V(axis_divisions * height_divisions + 1, 0) = 0;
    V(axis_divisions * height_divisions + 1, 1) = 0;
    V(axis_divisions * height_divisions + 1, 2) = height;

    //Bottom face
    for (int i = 0; i < axis_divisions; ) {
        F(f, 0) = axis_divisions * height_divisions;
        F(f, 1) = i % axis_divisions;
        ++i;
        F(f, 2) = i % axis_divisions;
        ++f;
    }

    //Top face
    for (int i = 0; i < axis_divisions; ) {
        F(f, 0) = axis_divisions * height_divisions + 1;
        F(f, 1) = i % (axis_divisions)+(height_divisions - 1) * axis_divisions;
        ++i;
        F(f, 2) = i % (axis_divisions)+(height_divisions - 1) * axis_divisions;
        ++f;
    }


    return { V, F };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_cuboid(double a, double b, double c) {
    Eigen::MatrixXd V = (Eigen::MatrixXd(8, 3) <<
        a * 0.5, -b * 0.5, c * 0.5,
        a * 0.5, b * 0.5, c * 0.5,
        -a * 0.5, -b * 0.5, c * 0.5,
        -a * 0.5, b * 0.5, c * 0.5,
        a * 0.5, -b * 0.5, -c * 0.5,
        a * 0.5, b * 0.5, -c * 0.5,
        -a * 0.5, -b * 0.5, -c * 0.5,
        -a * 0.5, b * 0.5, -c * 0.5).finished();
    Eigen::MatrixXi F = (Eigen::MatrixXi(12, 3) <<
        0, 6, 4,
        0, 2, 6,
        0, 3, 2,
        0, 1, 3,
        2, 7, 6,
        2, 3, 7,
        4, 6, 7,
        4, 7, 5,
        0, 4, 5,
        0, 5, 1,
        1, 5, 7,
        1, 7, 3).finished();

    return { V, F };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_ovoid(double ra, double rb, double rc) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    Tr tr;            // 3D-Delaunay triangulation
    C2t3 c2t3(tr);   // 2D-complex in 3D-Delaunay triangulation

    // defining the surface
    Surface_3 surface(sphere_function,             // pointer to function
        Sphere_3(CGAL::ORIGIN, 1.2)); // bounding sphere
    // Note that "2." above is the *squared* radius of the bounding sphere!
    // defining meshing criteria
    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(30.,  // angular bound
        0.1,  // radius bound
        0.05); // distance bound
// meshing surface
    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
    Surface_mesh sm;
    CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, sm);
    std::ofstream out("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/tempOvoid.off");
    out << sm << std::endl;
    std::cout << "Final number of mesh points: " << tr.number_of_vertices() << "\n";

    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/tempOvoid.off", V, F);

    ra /= 2;
    rb /= 2;
    rc /= 2;

    int scale = ra;
    for (int i = 0; i < V.rows(); i++) {
        V(i, 0) *= scale;
        V(i, 1) *= scale;
        V(i, 2) *= scale;
    }

    if (ra != rb || rb != rc || ra != rc) {
        double scale1 = rb / ra;
        double scale2 = rc / ra;
        for (int i = 0; i < V.rows(); i++) {
            V(i, 1) *= scale1;
            V(i, 2) *= scale2;
        }
    }

    return { V, F };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> generate_meshB() {
    Eigen::MatrixXd v1, v2;
    Eigen::MatrixXi f1, f2;

    /* Generate material B STL files at specified location*/
    double B_SIZE1, B_SIZE2, B_SIZE3;
    double A_SIZE = 0;
    double GAP_DISTANCE1 = 0, GAP_DISTANCE2 = 0, GAP_DISTANCE3 = 0;
    //std::tuple<MatrixXd, MatrixXi> basic_mesh;

    int exit_loop = 0;
    std::cout << "Select material B SHAPE (1 for ovoid,  2 for cylinder, 3 for cuboid): ";
    while (exit_loop == 0) {
        int B_TYPE;
        cin >> B_TYPE;
        switch (B_TYPE)
        {
        default:
            std::cout << "Wrong number! Please enter a VALID material B type: ";
            break;
        case 1: {
            std::cout << "Ovoid shape selected!" << std::endl;
            std::cout << "Please enter the three DIAMETER dimensions of ovoid (separate by space): ";
            cin >> B_SIZE1 >> B_SIZE2 >> B_SIZE3;
            std::cout << endl << "Generating material B...";
            std::tuple<MatrixXd, MatrixXi> basic_mesh = generate_ovoid(B_SIZE1, B_SIZE2, B_SIZE3);
            v1 = get<0>(basic_mesh);
            f1 = get<1>(basic_mesh);
            exit_loop = 1;
            break; }
        case 2: {
            std::cout << "Cylinder shape selected!" << std::endl;
            std::cout << "Please enter the Diameter and Height dimensions of cylinder (separate by space): ";
            cin >> B_SIZE1 >> B_SIZE2;
            std::cout << endl << "Generating material B...";
            std::tuple<MatrixXd, MatrixXi> basic_mesh = generate_cylinder(B_SIZE1, B_SIZE2);
            v1 = get<0>(basic_mesh);
            f1 = get<1>(basic_mesh);
            exit_loop = 1;
            break; }
        case 3: {
            std::cout << "Cuboid shape selected!" << std::endl;
            std::cout << "Please enter the three dimensions of cuboid (separate by space): ";
            cin >> B_SIZE1 >> B_SIZE2 >> B_SIZE3;
            std::cout << endl << "Generating material B..." << endl;
            std::tuple<MatrixXd, MatrixXi> basic_mesh = generate_cuboid(B_SIZE1, B_SIZE2, B_SIZE3);
            v1 = get<0>(basic_mesh);
            f1 = get<1>(basic_mesh);
            exit_loop = 1;
            break; }
        } // end of switch

    }

    string filename = "C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/tempBasicMesh.STL";
    igl::writeSTL(filename, v1, f1);
    CGAL_remesh(filename);
    igl::read_triangle_mesh(filename, v2, f2); // we are going to use v2

    std::cout << "Please enter the bounding box SIDE LENGTH press ENTER: ";
    cin >> A_SIZE;
    std::cout << "Please enter the THREE GAP DISTANCEs between material B and press ENTER: ";
    cin >> GAP_DISTANCE1 >> GAP_DISTANCE2 >> GAP_DISTANCE3;

    int B_number1 = A_SIZE / (B_SIZE1 + GAP_DISTANCE1);
    int B_number2 = A_SIZE / (B_SIZE2 + GAP_DISTANCE2);
    int B_number3 = A_SIZE / (B_SIZE3 + GAP_DISTANCE3);// number of material B at one edge
    std::cout << B_number1 * B_number2 * B_number3 << " of material B will be generated ..." << endl;
    std::cout << endl;

    // Create material B meshes now
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    // Linear material B pattern
    vector<double> space1 = linspace(-A_SIZE / 2, A_SIZE / 2, B_number1);
    vector<double> space2 = linspace(-A_SIZE / 2, A_SIZE / 2, B_number2);
    vector<double> space3 = linspace(-A_SIZE / 2, A_SIZE / 2, B_number3);

    for (int layer = 0; layer < B_number3; ++layer) {
        for (int i = 0; i < B_number2; ++i) {
            for (int j = 0; j < B_number1; ++j) {
                Eigen::MatrixXd v_tmp1 = move_mesh(v2, space1[j], space2[i], space3[layer]);
                j++;
                if (j < B_number1) {
                    Eigen::MatrixXd v_tmp2 = move_mesh(v2, space1[j], space2[i], space3[layer]);
                    std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f2, f2);
                    meshB.push_back(mesh_matrix);
                }
                else {
                    std::tuple<MatrixXd, MatrixXi> mesh_matrix = { v_tmp1, f2 };
                    meshB.push_back(mesh_matrix);
                }
            }
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

    return { get<0>(tmp1), get<1>(tmp1) };
    //igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB_TEST.stl", get<0>(tmp1), get<1>(tmp1));
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
    switch (key)
    {
    default:
        break;
    case '1': {
        MOVE_MESH(main_v1, 10, 0, 0);
        // Visualize the relative position of material A & B
        MatrixXd V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        V << main_v1, main_v2;
        MatrixXi F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(V, F);
        return true; }
    case '2': {
        MOVE_MESH(main_v1, -10, 0, 0);
        // Visualize the relative position of material A & B
        MatrixXd main_V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        main_V << main_v1, main_v2;
        MatrixXi main_F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        main_F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(main_V, main_F);
        return true; }
    case '3': {
        MOVE_MESH(main_v1, 0, 10, 0);
        // Visualize the relative position of material A & B
        MatrixXd V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        V << main_v1, main_v2;
        MatrixXi F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(V, F);
        return true; }
    case '4': {
        MOVE_MESH(main_v1, 0, -10, 0);
        // Visualize the relative position of material A & B
        MatrixXd main_V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        main_V << main_v1, main_v2;
        MatrixXi main_F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        main_F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(main_V, main_F);
        return true; }
    case '5': {
        MOVE_MESH(main_v1, 0, 0, 10);
        // Visualize the relative position of material A & B
        MatrixXd V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        V << main_v1, main_v2;
        MatrixXi F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(V, F);
        return true; }
    case '6': {
        MOVE_MESH(main_v1, 0, 0, -10);
        // Visualize the relative position of material A & B
        MatrixXd main_V(main_v1.rows() + main_v2.rows(), main_v1.cols());
        main_V << main_v1, main_v2;
        MatrixXi main_F(main_f1.rows() + main_f2.rows(), main_f1.cols());
        main_F << main_f1, (main_f2.array() + main_v1.rows());
        //viewer.data().clear();
        viewer.data().set_mesh(main_V, main_F);
        return true; }
    }
    return false;
}


// Poisson distribution sub-sampling on surface
void create_poisson_points() {
    Eigen::MatrixXd V, v1;
    Eigen::MatrixXi F, f1;

    string filename = "C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Large_ball_accurate";
    igl::read_triangle_mesh(filename + ".STL", V, F);
    igl::writeOFF(filename + ".OFF", V, F);
    std::cout << "vertices number: " << V.rows() << endl;

    std::vector<Point> Mesh_Points;
    CGAL::IO::read_points(filename + ".OFF", std::back_inserter(Mesh_Points));

    // Compute normals
    Mesh mesh;
    if (!PMP::IO::read_polygon_mesh(filename + ".OFF", mesh) || !CGAL::is_triangle_mesh(mesh))
    {
        std::cerr << "Invalid input." << std::endl;
        return;
    }
    std::map<vertex_descriptor, Vector_3> vnormals;
    PMP::compute_vertex_normals(mesh, boost::make_assoc_property_map(vnormals), PMP::parameters::vertex_point_map(mesh.points()).geom_traits(K()));

    // Poisson-disk Sampling
    // Copy vertices and normals information
    std::vector<Vec3> verts;
    std::vector<Vec3> nors;
    BOOST_FOREACH(vertex_descriptor vd, vertices(mesh)) {
        nors.push_back(Vec3(vnormals[vd][0], vnormals[vd][1], vnormals[vd][2]));
        //cout << "Normal: " << " " << vnormals[vd][0] << " " << vnormals[vd][1] << " " << vnormals[vd][2];
        //cout << " at vertex: " << mesh.point(vd)[0] << " " << mesh.point(vd)[1] << " " << mesh.point(vd)[2] << endl;
        verts.push_back(Vec3(mesh.point(vd)[0], mesh.point(vd)[1], mesh.point(vd)[2]));
    }

    std::vector<int> tris; // list of triangles and vertex ids
    BOOST_FOREACH(face_descriptor fd, faces(mesh)) {
        list<int> v_list;
        BOOST_FOREACH(halfedge_descriptor hd, halfedges_around_face(mesh.halfedge(fd), mesh)) {
            int vid1 = mesh.vertex(mesh.edge(hd), 1);
            int vid2 = mesh.vertex(mesh.edge(hd), 0);
            v_list.push_back(vid1);
            v_list.push_back(vid2);
        }
        auto it = v_list.begin();
        tris.push_back(*it);
        ++it;
        tris.push_back(*it);
        ++it;
        tris.push_back(*it);
    }

    // Store new sampled points and normals
    int sample_radius;
    std::cout << "Please enter sample distance: ";
    cin >> sample_radius;
    int sample_size = 0;
    double exposure_percentage = 50.0;
    std::cout << "Please enter material B exposure percentage: ";
    cin >> exposure_percentage;
    std::vector<Vec3> samples_pos;
    std::vector<Vec3> samples_nor;
    poisson_disk(sample_radius, sample_size, verts, nors, tris, samples_pos, samples_nor);

    std::vector<Point> output;
    //std::vector<Point> output = Mesh_Points;
    for (auto ele : samples_pos) {
        output.push_back(Point(ele.x, ele.y, ele.z));
    }
    // Compute average spacing using neighborhood of 6 points
    //double spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(output, 5);
    std::tuple<double, double, double, double, double> spacings = CGAL::WAAM_compute_average_spacing<CGAL::Sequential_tag>(Mesh_Points, 4);
    std::cout << "Minimum of the minimum spacing between a point and its nearest neighbor: " << get<0>(spacings) << endl;
    std::cout << "Maximum of the minimum spacing between a point and its nearest neighbor: " << get<1>(spacings) << endl;
    std::cout << "Average of the minimum spacing between a point and its nearest neighbor: " << get<2>(spacings) << endl;
    std::cout << "Median of the minimum spacing between a point and its nearest neighbor: " << get<3>(spacings) << endl;
    std::cout << "Standard deviation among all point-point distances: " << get<4>(spacings) << endl;
    //cout << "The average of all point-point distances: " << get<4>(spacings) << endl;


    // Now generate material B based on grid points
    int N = samples_pos.size();
    printf("Generating %i meshB elements... \n", N);
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_ball.stl", v1, f1);
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    //for (int i = 0; i < Mesh_Points.size() - 1; ++i) {
    //    Eigen::MatrixXd v_tmp1 = move_mesh(v1, Mesh_Points[i][0], Mesh_Points[i][1], Mesh_Points[i][2]);
    //    ++i;
    //    Eigen::MatrixXd v_tmp2 = move_mesh(v1, Mesh_Points[i][0], Mesh_Points[i][1], Mesh_Points[i][2]);

    //    std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
    //    meshB.push_back(mesh_matrix);
    //}

    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d Nvec1(samples_nor[i].x, samples_nor[i].y, samples_nor[i].z);
        Eigen::MatrixXd Rv1 = rotate_mesh(v1, Nvec1);
        //Eigen::MatrixXd v_tmp1 = move_mesh(v1, samples_pos[i].x, samples_pos[i].y, samples_pos[i].z);
        Eigen::MatrixXd v_tmp1 = move_mesh(Rv1, samples_pos[i].x + (samples_nor[i].x * (exposure_percentage - 50) / 100.0 * 40), samples_pos[i].y + (samples_nor[i].y * (exposure_percentage - 50) / 100.0 * 40), samples_pos[i].z + (samples_nor[i].z * (exposure_percentage - 50) / 100.0 * 40));

        if (i < N - 1) {
            ++i;
            Eigen::Vector3d Nvec2(samples_nor[i].x, samples_nor[i].y, samples_nor[i].z);
            Eigen::MatrixXd Rv2 = rotate_mesh(v1, Nvec2);
            //Eigen::MatrixXd v_tmp2 = move_mesh(v1, samples_pos[i].x, samples_pos[i].y, samples_pos[i].z);
            Eigen::MatrixXd v_tmp2 = move_mesh(Rv2, samples_pos[i].x + (samples_nor[i].x * (exposure_percentage - 50) / 100.0 * 40), samples_pos[i].y + (samples_nor[i].y * (exposure_percentage - 50) / 100.0 * 40), samples_pos[i].z + (samples_nor[i].z * (exposure_percentage - 50) / 100.0 * 40));

            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
        else {
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp1, f1, f1);
            meshB.push_back(mesh_matrix);
        }

    }

    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;
    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB__Distribution.stl", get<0>(tmp1), get<1>(tmp1));

    // Delete generated OFF file
    string removeName = filename + ".OFF";
    const char* removeNAME = removeName.c_str();
    remove(removeNAME);
}

// The input can also be vector<Point> points which is Sample_Points
void evaluate(vector<double> distances, double target_length) {
    //std::vector<Point> points;
    ////Point_set points;
    //CGAL::IO::read_points("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/pcd.ply", std::back_inserter(points));

    /*
    // Reconstruct a mesh surface using points and then evaluate edge length
    std::vector<Facet> facets;
    // The function is called using directly the points raw iterators
    CGAL::advancing_front_surface_reconstruction(points.begin(), points.end(),
        std::back_inserter(facets),
        2.0,
        (target_length * sqrt(1 + p_deviation)));

    std::cout << facets.size() << " facet(s) generated by reconstruction." << std::endl;

    CGAL::IO::write_STL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/tessellation.stl", points, facets);

    Mesh mesh;
    std::vector<double> distances;
    PMP::IO::read_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/tessellation.stl", mesh);

    BOOST_FOREACH(edge_descriptor e, edges(mesh))
    {
        halfedge_descriptor hd = halfedge(e, mesh);
        double length = CGAL::approximate_sqrt(CGAL::squared_distance(mesh.point(target(hd, mesh)), mesh.point(target(opposite(hd, mesh), mesh))));
        //if(length < target_length * sqrt(1 + p_deviation) && length > min_distance) distances.push_back(length);
        distances.push_back(length);
    }
    */

    /*
    std::vector<double> distances;
    unordered_map<Point, int> umap;

    // Find the number of neighbors for each point
    for (int i = 0; i < points.size(); ++i) {
        for (int j = 0; j < points.size(); ++j) {
            double d = sqrt(CGAL::squared_distance(points[i], points[j]));
            if (d < p_deviation && d > min_distance) {
                umap[points[i]]++;
                distances.push_back(d);
            }
        }
    }

    int count5 = 0;
    int count6 = 0;
    int count7 = 0;
    for (auto p : umap) {
        if (p.second == 5) count5++;
        else if (p.second == 6) count6++;
        else if (p.second == 7) count7++;
    }

    std::cout << "Percentage of points with 5 neighbors: " << (double)count5 * 100 / (double)points.size() << " %" << std::endl;
    std::cout << "Percentage of points with 6 neighbors: " << (double)count6 * 100 / (double)points.size() << " %" << std::endl;
    std::cout << "Percentage of points with 7 neighbors: " << (double)count7 * 100 / (double)points.size() << " %" << std::endl;
    */

    int d_size = distances.size();
    std::cout << "Error percentage        Percentage of point-point distances" << endl;

    std::vector<double> range_percentage(8, 0.0);
    int count = 1;
    int i = 0;
    int z = 0;
    int flag = 0;
    for (; i < d_size && count <= 8; ) {
        if (distances[i] > target_length * (0.8 + count * 0.05) && flag == 0) {
            count++;
            continue;
        }
        else if (distances[i] > target_length * (0.8 + (count - 1) * 0.05) && distances[i] < target_length * (0.8 + count * 0.05)) {
            if (i < d_size - 1) {
                flag = 1;
                i++;
                continue;
            }
            else {
                range_percentage[count - 1] = (double)(i - z) / (double)d_size * 100;
                break; // reach the end
            }
        }

        else if (distances[i] > target_length * (0.8 + count * 0.05) && flag != 0) {
            // 5% increment
            range_percentage[count - 1] = (double)(i - z) / (double)d_size * 100;
            //std::cout << "At index: " << i << " in this range no. of distances: " << (i - z) << " reach at distance: " << distances[i] << endl;
            z = i;
            count++;
            i++;
        }
    }

    for (int j = 1; j <= 8; ++j) {
        std::cout << (j - 4 - 1) * 5 << " - ";
        std::cout << (j - 4) * 5 << "%               ";

        printf("%.2lf", range_percentage[j - 1]);
        std::cout << "%" << endl;
    }


    //{
    //    // For poisson and other methods
    //    std::vector<double> range_percentage(21, 0.0);
    //    int count = 1;
    //    int z = 0;
    //    for (int i = 0; i < point_size && count <= 20; ++i) {
    //        if (distances[i] < target_length * (1 + count * 0.05)) // 5% increment
    //            continue;
    //        else {
    //            range_percentage[count - 1] = (double)(i - z) / (double)point_size * 100;
    //            z = i;
    //            count++;
    //        }
    //    }

    //    if (count == 21 && z < point_size) range_percentage[20] = (point_size - z) / (double)point_size * 100;

    //    for (int j = 1; j <= 20; ++j) {
    //        std::cout << (j - 1) * 5 << " - ";
    //        std::cout << j * 5 << "%               ";

    //        if (log10(j * 5) < 1) std::cout << "   ";
    //        else if (log10(j * 5) >= 1 && log10(j * 5) < 2 && log10((j - 1) * 5) < 1) std::cout << "  ";
    //        else if (log10(j * 5) >= 1 && log10(j * 5) < 2 && log10((j - 1) * 5) >= 1) std::cout << " ";

    //        printf("%.2lf", range_percentage[j - 1]);
    //        std::cout << "%" << endl;
    //        //cout << string(  std::floor(range_percentage[j - 1] * 50 / 100), '*') << endl;
    //    }

    //    std::cout << "100% and above:         ";
    //    printf("%.2lf", range_percentage[20]);
    //    std::cout << "%     ";
    //}

}

// Functions for generating uniform points
bool CheckNeighbor(Point candidate, vector<Point> PVect, double distance) {
    //std::cout << "search intersection point: " << (candidate)[0] << " " << (candidate)[1] << " " << (candidate)[2] << endl;
    for (auto it = PVect.begin(); it != PVect.end(); ++it) {
        //std::cout << "checking point: " << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << endl;
        if (candidate == *it) {
            return false;
        }
        else if (CGAL::squared_distance(*it, candidate) < distance * distance) {
            //std::cout << "Too close with point: " << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << endl;
            return false;
        }
        else continue;
    }

    return true;
}

// If null point returned, then it is case C, otherwise case B
Point CheckNearPoint(Point P_new, vector<Point> Sample_Points, double deviation) {
    for (auto it = Sample_Points.begin(); it != Sample_Points.end(); ++it) {
        if (CGAL::squared_distance(*it, P_new) < deviation * deviation) {
            return *it;
        }
    }
    return null_point;
}

// For growing front meet boundary points
Point CheckBoundaryPoint(Point P_new, vector<Point> Sample_Points, double deviation1, double deviation2) {
    for (auto it = Sample_Points.begin(); it != Sample_Points.end(); ++it) {
        if (CGAL::squared_distance(*it, P_new) < deviation2 * deviation2 && CGAL::squared_distance(*it, P_new) > deviation1 * deviation1) {
            return *it;
        }
    }
    return null_point;
}

// Compute the area of a triangle given its three vertices
double triangle_area(const Point_3& p1, const Point_3& p2, const Point_3& p3) {
    Vector_3 v1 = p2 - p1;
    Vector_3 v2 = p3 - p1;
    return std::sqrt(cross_product(v1, v2).squared_length()) / 2;
}

// Compute the surface area of a mesh
double mesh_surface_area(const Surface_mesh& mesh) {

    double area = 0.0;
    Surface_mesh::Face_range faces = mesh.faces();

    for (Surface_mesh::Face_index f : faces) {
        list<Mesh::Vertex_index> v_list;
        BOOST_FOREACH(halfedge_descriptor hd, halfedges_around_face(mesh.halfedge(f), mesh)) {
            Mesh::Vertex_index vid1 = mesh.vertex(mesh.edge(hd), 1);
            Mesh::Vertex_index vid2 = mesh.vertex(mesh.edge(hd), 0);
            v_list.push_back(vid1);
            v_list.push_back(vid2);
        }

        // Get the three vertices of the current face
        auto vbegin = v_list.begin();
        Point_3 p1 = mesh.point(*vbegin++);
        Point_3 p2 = mesh.point(*vbegin++);
        Point_3 p3 = mesh.point(*vbegin);
        area += triangle_area(p1, p2, p3);
    }
    return area;
}

bool checkFace(Mesh mesh, Point p1, Point p2, Point p3) {
    for (auto face : mesh.faces()) {
        auto vrange = mesh.vertices_around_face(mesh.halfedge(face));
        auto v1 = mesh.point(*(vrange.begin()));
        auto v2 = mesh.point(*(++vrange.begin()));
        auto v3 = mesh.point(*(++(++vrange.begin())));
        if ((v1 == p1 && v2 == p2 && v3 == p3) || (v1 == p1 && v2 == p3 && v3 == p2) || (v1 == p2 && v2 == p3 && v3 == p1)
            || (v1 == p2 && v2 == p1 && v3 == p3) || (v1 == p3 && v2 == p1 && v3 == p2) || (v1 == p3 && v2 == p2 && v3 == p1))
        {
            return false;
        }
        else continue;
    }

    return true;
}

std::vector<pair<Edge, Edge>> findConnectEdge(unordered_set<pair<Point, Point>, pair_hash> edge_around1, unordered_set<pair<Point, Point>, pair_hash> edge_around2) {
    vector<pair<Edge, Edge>> edge_vect;
    for (auto pair1 : edge_around1) { // these are the new edges
        for (auto pair2 : edge_around2) {
            if (pair1.first == pair2.first || pair1.first == pair2.second || pair1.second == pair2.first || pair1.second == pair2.second) {
                edge_vect.push_back(make_pair(Edge(pair1.first, pair1.second), Edge(pair2.first, pair2.second)));
                //std::cout << "added set: " << pair1.first << " " << pair1.second << "    and " << pair2.first << " " << pair2.second << endl;
            }
            else continue;
        }
    }

    //std::cout << "Connected edges set size: " << edge_vect.size() << endl;
    return edge_vect; // should always find two new connected locations
}

// relax for the first 12 points and 2 cells
void initial_relax_points(vector<Point>& relax_group, Tree& tree, Point P_SLOW, Point P_FAST, double r) {

    K_neighbor_search search0(tree, relax_group[0], 6); // search for two circles of nearby points 6+12
    K_neighbor_search search1(tree, relax_group[1], 6);
    K_neighbor_search search2(tree, relax_group[2], 6);
    K_neighbor_search search3(tree, relax_group[3], 6);

    double Min_sum = pow(sqrt(CGAL::squared_distance(relax_group[0], P_SLOW)) - r, 2) + pow(sqrt(CGAL::squared_distance(relax_group[0], relax_group[1])) - r, 2) +
        pow(sqrt(CGAL::squared_distance(relax_group[1], relax_group[2])) - r, 2) + pow(sqrt(CGAL::squared_distance(relax_group[2], relax_group[3])) - r, 2)
        + pow(sqrt(CGAL::squared_distance(relax_group[3], P_FAST)) - r, 2);

    for (K_neighbor_search::iterator it0 = search0.begin(); it0 != search0.end(); ++it0)
    {
        for (K_neighbor_search::iterator it1 = search1.begin(); it1 != search1.end(); ++it1) {
            for (K_neighbor_search::iterator it2 = search2.begin(); it2 != search2.end(); ++it2) {
                for (K_neighbor_search::iterator it3 = search3.begin(); it3 != search3.end(); ++it3) {
                    double d = pow(sqrt(CGAL::squared_distance(it0->first, P_SLOW)) - r, 2) + pow(sqrt(CGAL::squared_distance(it0->first, it1->first)) - r, 2) +
                        pow(sqrt(CGAL::squared_distance(it1->first, it2->first)) - r, 2) + pow(sqrt(CGAL::squared_distance(it2->first, it3->first)) - r, 2)
                        + pow(sqrt(CGAL::squared_distance(it3->first, P_FAST)) - r, 2);
                    if (d < Min_sum) {
                        Min_sum = d;
                        relax_group[0] = it0->first;
                        relax_group[1] = it1->first;
                        relax_group[2] = it2->first;
                        relax_group[3] = it3->first;
                    }
                }
            }
        }
    } // end of for loop
}

// relax for all other cells
// only two or three surrounding points
void relax_points(vector<Point>& relax_group, Tree& tree, Point p1, Point p3, Point P_center, double r) {

    K_neighbor_search search0(tree, relax_group[0], 12);
    K_neighbor_search search1(tree, relax_group[1], 12);
    K_neighbor_search search2(tree, relax_group[2], 12);

    double Min_sum = pow(sqrt(CGAL::squared_distance(relax_group[0], p3)) - r, 2) + pow(sqrt(CGAL::squared_distance(relax_group[0], relax_group[1])) - r, 2) +
        pow(sqrt(CGAL::squared_distance(relax_group[1], relax_group[2])) - r, 2) + pow(sqrt(CGAL::squared_distance(relax_group[2], p1)) - r, 2) +
        pow(sqrt(CGAL::squared_distance(relax_group[0], P_center)) - r, 2) + pow(sqrt(CGAL::squared_distance(relax_group[1], P_center)) - r, 2) +
        pow(sqrt(CGAL::squared_distance(relax_group[2], P_center)) - r, 2);

    //std::cout << "Initial deviation " << Min_sum << endl << endl;
    for (K_neighbor_search::iterator it0 = search0.begin(); it0 != search0.end(); ++it0)
    {
        for (K_neighbor_search::iterator it1 = search1.begin(); it1 != search1.end(); ++it1) {
            for (K_neighbor_search::iterator it2 = search2.begin(); it2 != search2.end(); ++it2) {
                double d = pow(sqrt(CGAL::squared_distance(it0->first, p3)) - r, 2) + pow(sqrt(CGAL::squared_distance(it0->first, it1->first)) - r, 2) +
                    pow(sqrt(CGAL::squared_distance(it1->first, it2->first)) - r, 2) + pow(sqrt(CGAL::squared_distance(it2->first, p1)) - r, 2) +
                    pow(sqrt(CGAL::squared_distance(it1->first, P_center)) - r, 2) + pow(sqrt(CGAL::squared_distance(it2->first, P_center)) - r, 2) +
                    pow(sqrt(CGAL::squared_distance(it0->first, P_center)) - r, 2);
                if (d < Min_sum) {
                    Min_sum = d;
                    relax_group[0] = it0->first;
                    relax_group[1] = it1->first;
                    relax_group[2] = it2->first;
                }
            }
        }
    }
    //std::cout << "Relaxed deviation " << Min_sum << endl;
    //std::cout << "Relaxed outside edge length: " << sqrt(CGAL::squared_distance(relax_group[0], p3)) << endl;
    //std::cout << "Relaxed outside edge length: " << sqrt(CGAL::squared_distance(relax_group[0], relax_group[1])) << endl;
    //std::cout << "Relaxed outside edge length: " << sqrt(CGAL::squared_distance(relax_group[1], relax_group[2])) << endl;
    //std::cout << "Relaxed outside edge length: " << sqrt(CGAL::squared_distance(relax_group[2], p1)) << endl;
}

double find_height(unordered_set<double> height_set, Point p, double d) {
    for (auto ele : height_set) {
        if (abs(p[1] - ele) < d) return ele;
    }

    return 0.0;
}

// Collect border points and shrink it based on user specifications
#include <CGAL/Polygon_mesh_processing/border.h>
bool check_border(Point candidate, std::set<Point> border_points) {
    if (border_points.find(candidate) == border_points.end()) return true;
    else return false;
}


// For cylindrical surfaces
vector<Point> create_uniform_points(Mesh mesh, double t, double r, double deviation) {

    vector<Point> Sample_Points;
    vector<Point> relax_group; // usually 4 points
    unordered_set<Point> boundary_points;
    unordered_set<double> height_set;
    double min_distance = 0.0;
    double target_distance_p = t * (1 + deviation / 100.0);
    double target_distance_n = (t * (1 - deviation / 100.0) > min_distance) ? t * (1 - deviation / 100.0) : min_distance;
    double near_point_threshold = (target_distance_p - target_distance_n);
    double angle_prev = 0.0;
    //cout << "lower bound: " << target_distance_n << " and upper bound: " << target_distance_p << endl;

    Mesh out_mesh; // Sample points mesh
    queue<pair<Point, Point>> edge_queue;
    unordered_set<pair<Point, Point>, pair_hash> edge_around1;
    unordered_set<pair<Point, Point>, pair_hash> edge_around2;
    Point p_slow, p_fast; // p_slow is the starting point

    // Clear point too close to the border
    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));

    std::set<vertex_descriptor> border_points;
    for (auto b : border) {
        border_points.insert(source(b, mesh));
        border_points.insert(target(b, mesh));
    }


    double remove_distance = r / 2 * 1.1; // can be a user input


    // Delete points within the distance threshold from the border
    std::set<Point> deletedPoints;
    // Iterate over all points in the mesh
    for (vertex_descriptor v : vertices(mesh))
    {
        // Check the distance from the point to the border points
        for (vertex_descriptor borderV : border_points)
        {
            if (borderV == v) continue;

            if (CGAL::squared_distance(mesh.point(borderV), mesh.point(v)) < remove_distance * remove_distance)
            {
                deletedPoints.insert(mesh.point(v));
                break;
            }
        }
    } // Done with the for loop
    //std::cout << "deletedPoints size: " << deletedPoints.size() << endl;

    // Insert all mesh points in the kd-tree
    Tree tree(mesh.points().begin(), mesh.points().end(), Splitter());

    // Select the starting point from the centroid
    Point p_centroid = CGAL::centroid(mesh.points().begin(), mesh.points().end(), CGAL::Dimension_tag<0>());
    double minDistance = INT32_MAX;
    for (const auto& v : mesh.vertices()) {
        double distance = sqrt(CGAL::squared_distance(mesh.point(v), p_centroid));
        if (distance < minDistance) {
            minDistance = distance;
            p_slow = mesh.point(v);
        }
    }

    //Point query = Point(p_slow[0], p_slow[1] - 3, p_slow[2]); // Manual adjust starting point
    //K_neighbor_search start_search(tree, query, 10);
    //p_slow = start_search.begin()->first;
    std::cout << "start point location: " << p_slow << endl;

    // Done selecting

    std::vector<Point> group1;
    std::vector<Point> group2;

    // Initial search around Point 1 for Point 2
    double area = mesh_surface_area(mesh);
    double area1 = 0;
    //cout << "Surface area: " << area << endl;
    //cout << "vertices number: " << std::distance(mesh.vertices_begin(), mesh.vertices_end()) << endl;
    int K = (PI * r * r) / area * std::distance(mesh.vertices_begin(), mesh.vertices_end()) * 1; // be careful about the multiplier 5
    std::cout << "Defined k number: " << K << endl;
    K_neighbor_search search3(tree, p_slow, K);
    // Find group1
    for (K_neighbor_search::iterator it = search3.begin(); it != search3.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
            group1.push_back(it->first);
        }
    }

    if (group1.size() == 0) {
        std::cout << "Enter a larger multiplier for K number!" << std::endl;
        return Sample_Points;
    }

    // Select the best candidate Point 2 around Point 1
    std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> initial_pq; // distance queue
    for (Point p_it : group1) {
        if (abs(p_it[1] - p_slow[1]) < 0.05) {
            double d = abs(sqrt(CGAL::squared_distance(p_it, p_slow)) - r); // sorted by how close to target distance
            initial_pq.emplace(d, p_it);
        }
    }

    while (!initial_pq.empty()) {
        if (check_border(std::get<1>(initial_pq.top()), deletedPoints)) {
            p_fast = std::get<1>(initial_pq.top()); //  the second point
            break;
        }
        else initial_pq.pop();
    }

    //std::cout << "Initial edge length: " << sqrt(CGAL::squared_distance(p_fast, p_slow)) << endl;
    Sample_Points.push_back(p_slow);
    Sample_Points.push_back(p_fast);
    group1.clear();
    //std::cout << "p_slow: " << p_slow << "     and p_fast: " << p_fast << endl;

    // Find next 4 edges outside the while loop
    K_neighbor_search search2(tree, p_slow, K);
    // Find group1
    for (K_neighbor_search::iterator it = search2.begin(); it != search2.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
            group1.push_back(it->first);
        }
    }

    // Find group2
    K_neighbor_search search1(tree, p_fast, K);
    for (K_neighbor_search::iterator it = search1.begin(); it != search1.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
            group2.push_back(it->first);
        }
    }

    // Select Point 3&4 from the intersection group points
    std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> second_pq; // distance queue
    for (Point it1 : group1) {
        for (Point it2 : group2) {
            if (it1 == it2) {
                double d = pow((sqrt(CGAL::squared_distance(it1, p_slow)) - r), 2) + pow((sqrt(CGAL::squared_distance(it1, p_fast)) - r), 2);
                second_pq.emplace(d, it1);
            }
        }
    }

    // Find point 3
    while (!second_pq.empty()) {
        if (check_border(std::get<1>(second_pq.top()), deletedPoints)) {
            break;
        }
        else second_pq.pop();
    }
    Point P1 = std::get<1>(second_pq.top());
    Point P2;
    Point P_SLOW = p_slow;
    Point P_FAST = p_fast;
    Sample_Points.push_back(P1);
    second_pq.pop();
    // Find point 4
    while (!second_pq.empty()) { // see if a new point can be found
        if (CheckNeighbor(std::get<1>(second_pq.top()), Sample_Points, target_distance_n) && check_border(std::get<1>(second_pq.top()), deletedPoints)) {
            P2 = std::get<1>(second_pq.top());
            break; // point 4 is found
        }
        else second_pq.pop(); // continue searching
    }

    Point P_center = P1;
    edge_queue.push(make_pair(P_center, p_slow));

    // Add faces to out_mesh
    if (CGAL::orientationC3(p_slow[0], p_fast[0], P1[0], p_slow[1], p_fast[1], P1[1], p_slow[2], p_fast[2], P1[2]) == CGAL::LEFT_TURN)
        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P1));
    else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P1));

    if (CGAL::orientationC3(p_slow[0], p_fast[0], P2[0], p_slow[1], p_fast[1], P2[1], p_slow[2], p_fast[2], P2[2]) == CGAL::LEFT_TURN)
        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P2));
    else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P2));

    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P1));
    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P2));
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_fast, P1)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_fast, P2)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_slow, P1)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_slow, P2)) << endl;
    //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;

    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    Point P_prev = p_fast;
    {
        while (edge_queue.size() > 0) {
            // Clear group search points and update p_slow p_fast by extracting an edge
            group1.clear();
            group2.clear();


            p_slow = edge_queue.front().second;
            p_fast = P_center;

            // Find tow new edges and one candidate point, usually always can find one
            K_neighbor_search search_slow(tree, p_slow, K);
            // Find group1
            for (K_neighbor_search::iterator it = search_slow.begin(); it != search_slow.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
                    group1.push_back(it->first);
                }
            }

            // Find group2
            K_neighbor_search search_fast(tree, p_fast, K);
            for (K_neighbor_search::iterator it = search_fast.begin(); it != search_fast.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
                    group2.push_back(it->first);
                }
            }

            // Get intersection group points
            std::vector<Point> intersection_vect;
            for (Point it1 : group1) {
                for (Point it2 : group2) {
                    if (it1 == it2) {
                        intersection_vect.push_back(it1);
                    }
                }
            }


            std::vector<Point> p_candidate;
            for (Point p : intersection_vect) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) p_candidate.push_back(p);
            }
            //std::cout << "Candidate size: " << p_candidate.size() << endl;

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq; // distance queue
            //std::vector<Point> p_candidate = flag_it == 2 ? p2_vect : p1_vect;
            for (auto p_it : p_candidate) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 2); // sorted by how close to target distance
                pq.emplace(d, p_it);
            }

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_copy = pq;

            while (!pq.empty()) {
                // Can always find a new point in closed surface
                if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && abs(CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) - 120) < 10
                    && check_border(std::get<1>(pq.top()), deletedPoints)) // case A
                {
                    Point P_new = std::get<1>(pq.top()); // Find the valid new point
                    //std::cout << "New formed outside edge length: " << sqrt(CGAL::squared_distance(P_new, p_slow)) << endl;
                    //std::cout << "Internal angle: " << CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) << endl;

                    edge_queue.push(make_pair(P_center, P_new));

                    Sample_Points.push_back(P_new); // Only add point in case A
                    relax_group.push_back(P_new);

                    //area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_new));*/
                    //std::cout << "Added area: " << sqrt(CGAL::squared_area(p_slow, p_fast, P_new)) << endl;
                    break;
                }
                else pq.pop(); // search pq for the next candidate point
            } // End of pq while loop

            if (pq.empty()) { //check case B
                // initial 12 points
            }

            edge_queue.pop(); // Done with the current edge
            P_prev = p_slow;

            if (find(Sample_Points.begin(), Sample_Points.end(), P2) == Sample_Points.end() && edge_queue.size() == 0) {

                initial_relax_points(relax_group, tree, P_SLOW, P_FAST, r);
                int n = Sample_Points.size();
                Sample_Points[n - 4] = relax_group[0];
                Sample_Points[n - 3] = relax_group[1];
                Sample_Points[n - 2] = relax_group[2];
                Sample_Points[n - 1] = relax_group[3];
                edge_around1.insert(make_pair(P_SLOW, relax_group[0]));
                edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around1.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around1.insert(make_pair(relax_group[2], relax_group[3]));
                edge_around1.insert(make_pair(relax_group[3], P_FAST));
                relax_group.clear();

                Sample_Points.push_back(P2);
                P_center = P2;
                edge_queue.push(make_pair(P_center, P_SLOW));
                P_prev = P_FAST;

                if (CGAL::orientationC3(Sample_Points[n - 4][0], P_SLOW[0], P1[0], Sample_Points[n - 4][1], P_SLOW[1], P1[1], Sample_Points[n - 4][2], P_SLOW[2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], Sample_Points[n - 3][0], P1[0], Sample_Points[n - 4][1], Sample_Points[n - 3][1], P1[1], Sample_Points[n - 4][2], Sample_Points[n - 3][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 3][0], P1[0], Sample_Points[n - 2][1], Sample_Points[n - 3][1], P1[1], Sample_Points[n - 2][2], Sample_Points[n - 3][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 1][0], P1[0], Sample_Points[n - 2][1], Sample_Points[n - 1][1], P1[1], Sample_Points[n - 2][2], Sample_Points[n - 1][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 1][0], P_FAST[0], P1[0], Sample_Points[n - 1][1], P_FAST[1], P1[1], Sample_Points[n - 1][2], P_FAST[2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P1));
            }
            else if (edge_queue.size() == 0) {
                initial_relax_points(relax_group, tree, P_SLOW, P_FAST, r);
                int n = Sample_Points.size();
                Sample_Points[n - 4] = relax_group[0];
                Sample_Points[n - 3] = relax_group[1];
                Sample_Points[n - 2] = relax_group[2];
                Sample_Points[n - 1] = relax_group[3];
                edge_around2.insert(make_pair(P_SLOW, relax_group[0]));
                edge_around2.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around2.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around2.insert(make_pair(relax_group[2], relax_group[3]));
                edge_around2.insert(make_pair(relax_group[3], P_FAST));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], P_SLOW[0], P2[0], Sample_Points[n - 4][1], P_SLOW[1], P2[1], Sample_Points[n - 4][2], P_SLOW[2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], Sample_Points[n - 3][0], P2[0], Sample_Points[n - 4][1], Sample_Points[n - 3][1], P2[1], Sample_Points[n - 4][2], Sample_Points[n - 3][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 3][0], P2[0], Sample_Points[n - 2][1], Sample_Points[n - 3][1], P2[1], Sample_Points[n - 2][2], Sample_Points[n - 3][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 1][0], P2[0], Sample_Points[n - 2][1], Sample_Points[n - 1][1], P2[1], Sample_Points[n - 2][2], Sample_Points[n - 1][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 1][0], P_FAST[0], P2[0], Sample_Points[n - 1][1], P_FAST[1], P2[1], Sample_Points[n - 1][2], P_FAST[2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P2));
            }
            //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;
        } // End of edge_queue while loop
    }

    // Generate the first 12 initial points and two cells
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////

    vector<pair<Edge, Edge>> Connected_edges = findConnectEdge(edge_around1, edge_around2); // Edge is a data structure I defined
    queue<Edge> pentagon_queue;
    edge_around2.insert(edge_around1.begin(), edge_around1.end());
    edge_around1.clear();

    queue<pair<Edge, Edge>> Connected_edge_queue;
    for (auto e : Connected_edges) {
        Connected_edge_queue.push(e);
        edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
        edge_around2.erase(make_pair(e.second.source, e.second.target));
    }


    vector<Point> g1, g2, g3;
    vector<Point> flag5_vect;
    int start_point_flag = 3; // 3 means start with p3, otherwise maybe p1
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    while (Connected_edge_queue.size() > 0) {

        g1.clear();
        g2.clear();
        g3.clear();
        relax_group.clear();
        flag5_vect.clear();

        Edge edge1 = Connected_edge_queue.front().first;
        Edge edge2 = Connected_edge_queue.front().second;

        Point p1, p2, p3; // p2 is the middle point
        if (edge1.source == edge2.source || edge1.source == edge2.target) {
            p2 = edge1.source;
            p1 = edge1.target;
            p3 = edge1.source == edge2.source ? edge2.target : edge2.source;
        }
        else {
            p2 = edge1.target;
            p1 = edge1.source;
            p3 = edge1.target == edge2.source ? edge2.target : edge2.source;
        }
        //std::cout << "p1 location: " << p1 << "     p2 location: " << p2 << "     p3 location: " << p3 << endl;
        //std::cout << "p123 angle: " << CGAL::approximate_angle(p1, p2, p3) << endl;
        if (sqrt(CGAL::squared_distance(p1, p2)) > target_distance_p || sqrt(CGAL::squared_distance(p2, p3)) > target_distance_p
            || sqrt(CGAL::squared_distance(p1, p2)) < target_distance_n || sqrt(CGAL::squared_distance(p2, p3)) < target_distance_n) { // invalid edge pair
            Connected_edge_queue.pop();
            continue;
        }
        edge_around2.erase(make_pair(p1, p2));
        edge_around2.erase(make_pair(p2, p1));
        edge_around2.erase(make_pair(p3, p2));
        edge_around2.erase(make_pair(p2, p3));

        // Find a new center point to generate a cell around
        {
            // p1
            K_neighbor_search search_p1(tree, p1, K);
            for (K_neighbor_search::iterator it = search_p1.begin(); it != search_p1.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p1) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p1) >= pow(target_distance_n, 2)) {
                    g1.push_back(it->first);
                }
            }

            // p2
            K_neighbor_search search_p2(tree, p3, K);
            for (K_neighbor_search::iterator it = search_p2.begin(); it != search_p2.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p3) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p3) >= pow(target_distance_n, 2)) {
                    g2.push_back(it->first);
                }
            }
            // p3
            K_neighbor_search search_p3(tree, p3, K);
            for (K_neighbor_search::iterator it = search_p3.begin(); it != search_p3.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p3) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p3) >= pow(target_distance_n, 2)) {
                    g3.push_back(it->first);
                }
            }

            std::vector<Point> center_vect;
            std::vector<Point> center_vect_final;
            std::sort(g1.begin(), g1.end());
            std::sort(g2.begin(), g2.end());
            std::sort(g3.begin(), g3.end());
            std::set_intersection(g1.begin(), g1.end(), g2.begin(), g2.end(), std::back_inserter(center_vect));
            std::set_intersection(center_vect.begin(), center_vect.end(), g3.begin(), g3.end(), std::back_inserter(center_vect_final));

            std::vector<Point> pq_vect;
            for (Point p : center_vect_final) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) pq_vect.push_back(p);
            }
            //std::cout << "Center point candidate size: " << pq_vect.size() << endl;

            if (pq_vect.size() == 0) // need to generate a pentagon
            {
                if (center_vect_final.size() > 0) { // empty region when growing front meets
                    Point P_exist = null_point;
                    for (Point p : center_vect_final) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "boundary existing point found" << endl;
                            break;
                        }
                    }
                    if (P_exist == null_point) {
                        boundary_points.insert(p1);
                        boundary_points.insert(p2);
                        boundary_points.insert(p3);
                        //std::cout << "boundary point due to no new center" << endl;
                    }
                }

                //std::cout << "No new center point, pentagon case" << endl;
                queue<Edge> copy_queue = pentagon_queue;
                Point P_penta1, P_penta2; // each from a hexagon cell, used to find the new pentagon point
                while (!copy_queue.empty()) {

                    if (copy_queue.front().source == p1 || copy_queue.front().source == p3)
                    {
                        if (sqrt(CGAL::squared_distance(copy_queue.front().source, copy_queue.front().target)) < target_distance_p) {
                            P_penta1 = copy_queue.front().target;
                            P_penta2 = copy_queue.front().source == p1 ? p3 : p1;
                            //std::cout << "Penta point: " << P_penta1 << "   " << P_penta2 << endl;
                            break;
                        }
                        else copy_queue.pop();
                    }
                    else if (copy_queue.front().target == p1 || copy_queue.front().target == p3) {
                        if (sqrt(CGAL::squared_distance(copy_queue.front().source, copy_queue.front().target)) < target_distance_p) {
                            P_penta1 = copy_queue.front().source;
                            P_penta2 = copy_queue.front().source == p1 ? p3 : p1;
                            //std::cout << "Penta point: " << P_penta1 << "   " << P_penta2 << endl;
                            break;
                        }
                        else copy_queue.pop();
                    }
                    else copy_queue.pop();
                }

                K_neighbor_search search_penta1(tree, P_penta1, K);
                K_neighbor_search search_penta2(tree, P_penta2, K);
                std::vector<Point> penta_intersect;
                for (K_neighbor_search::iterator it1 = search_penta1.begin(); it1 != search_penta1.end(); it1++)
                {
                    if (CGAL::squared_distance(it1->first, P_penta1) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it1->first, P_penta1) >= pow(target_distance_n, 2)) {
                        for (K_neighbor_search::iterator it2 = search_penta2.begin(); it2 != search_penta2.end(); it2++)
                        {
                            if (CGAL::squared_distance(it2->first, P_penta2) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it2->first, P_penta2) >= pow(target_distance_n, 2))
                            {
                                if (it1->first == it2->first && CheckNeighbor(it1->first, Sample_Points, target_distance_n)) penta_intersect.push_back(it1->first);
                            }
                            else continue;
                        }
                    }
                    else continue;
                }

                if (penta_intersect.size() == 0) {
                    //std::cout << "No pentagon intersection points" << endl << endl;
                    edge_around1.insert(make_pair(P_penta2, P_penta1));
                    Connected_edges.clear();
                    Connected_edges = findConnectEdge(edge_around1, edge_around2);
                    edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
                    edge_around1.clear();

                    for (auto e : Connected_edges) {
                        //std::cout << "added pentagon edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
                        Connected_edge_queue.push(e);
                        edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
                        edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
                    }
                    Connected_edge_queue.pop();
                    continue; // go to the next connected edge pair
                }

                std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_penta;
                for (auto p_it : penta_intersect) {
                    double d = pow(sqrt(CGAL::squared_distance(p_it, P_penta1)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, P_penta2)) - r, 2);
                    pq_penta.emplace(d, p_it);
                }

                Point P_new = std::get<1>(pq_penta.top()); // insert a pentagon point
                Sample_Points.push_back(P_new);
                //std::cout << "New pentagon point: " << P_new << endl << endl;
                edge_around1.insert(make_pair(P_new, P_penta1));
                edge_around1.insert(make_pair(P_new, P_penta2));
                Connected_edges.clear();
                Connected_edges = findConnectEdge(edge_around1, edge_around2);
                edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
                edge_around1.clear();

                for (auto e : Connected_edges) {
                    //std::cout << "added pentagon edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
                    Connected_edge_queue.push(e);
                    edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
                    edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
                }

                Connected_edge_queue.pop();
                continue; // go to the next connected edge pair
            } // end of dealing with pentagon case

            // Now a new center point is found
            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_center; // distance queue
            for (auto p_it : pq_vect) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p1)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p2)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p3)) - r, 2);
                //double d = pow(sqrt(CGAL::squared_distance(p_it, p2)) - r, 2);
                pq_center.emplace(d, p_it);
            }

            if (CheckNeighbor(std::get<1>(pq_center.top()), Sample_Points, target_distance_n) && check_border(std::get<1>(pq_center.top()), deletedPoints)) {
                P_center = std::get<1>(pq_center.top());
                Sample_Points.push_back(P_center);
                //std::cout << "New center point: " << P_center << endl;
            }
            else {
                Connected_edge_queue.pop();
                continue;
            }
        }

        // Add two faces and three edges defined by connected two edges
        edge_queue.push(make_pair(P_center, p3));
        start_point_flag = 3;

        if (CGAL::orientationC3(p1[0], p2[0], P_center[0], p1[1], p2[1], P_center[1], p1[2], p2[2], P_center[2]) == CGAL::LEFT_TURN)
            out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_center));
        else out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p2));

        if (CGAL::orientationC3(p3[0], p2[0], P_center[0], p3[1], p2[1], P_center[1], p3[2], p2[2], P_center[2]) == CGAL::LEFT_TURN)
            out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_center));
        else out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p2));

        P_prev = p2;

        // Now generate a new cell around P_center
        while (edge_queue.size() > 0) {
            group1.clear();
            group2.clear();
            //pq_copy.swap(pq_empty); // clear the pq_copy by swapping with an empty queue

            p_slow = edge_queue.front().second;
            p_fast = P_center;


            // Find tow new edges and one candidate point, usually always can find one
            K_neighbor_search search_slow(tree, p_slow, K);
            // Find group1
            for (K_neighbor_search::iterator it = search_slow.begin(); it != search_slow.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
                    group1.push_back(it->first);
                }
            }

            // Find group2
            K_neighbor_search search_fast(tree, p_fast, K);
            for (K_neighbor_search::iterator it = search_fast.begin(); it != search_fast.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
                    group2.push_back(it->first);
                }
            }

            // Get intersection group points
            std::vector<Point> intersection_vect;
            for (Point it1 : group1) {
                for (Point it2 : group2) {
                    if (it1 == it2) {
                        intersection_vect.push_back(it1);
                    }
                }
            }

            std::vector<Point> p_candidate;
            for (Point p : intersection_vect) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) p_candidate.push_back(p);
            }

            //if (p_candidate.empty() && !intersection_vect.empty()) { // either there is an existing point or front boundary or no points at all
            //    Point P_exist = null_point;
            //    for (Point p : intersection_vect) {
            //        if (CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n) != null_point
            //            && sqrt(CGAL::squared_distance(p, P_prev)) > target_distance_n
            //            && find(relax_group.begin(), relax_group.end(), CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n)) == relax_group.end()
            //            && relax_group.size() < 2) { // case C

            //            if (start_point_flag == 1 && sqrt(CGAL::squared_distance(p, p1)) > target_distance_p) {
            //                P_exist = CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n);
            //                std::cout << "insert boundary point as existing point found: " << P_exist << endl;
            //                std::cout << "inserted: " << p_slow << endl;
            //                boundary_points.insert(p_slow);
            //                break;
            //            }
            //            else if (start_point_flag == 3 && relax_group.size() == 1) {
            //                std::cout << "inserted: " << p_slow << endl;
            //                boundary_points.insert(p_slow);
            //            }
            //            
            //        }
            //    }
            //}

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq; // distance queue
            //std::vector<Point> p_candidate = flag_it == 2 ? p2_vect : p1_vect;
            for (auto p_it : p_candidate) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 4) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 4); // sorted by how close to target distance
                //double d = 1.0/r/r/2.0*(pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 2)) + 1.0/120.0*abs(CGAL::approximate_angle(P_prev, p_slow, p_it) - 120);
                pq.emplace(d, p_it);
            }

            //std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_copy = pq; // copy the queue for later use

            while (!pq.empty()) {
                // Can always find a new point in closed surface
                // if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && abs(CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) - 120) < 15) // case A
                if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) < 123
                    && CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) > 117 && check_border(std::get<1>(pq.top()), deletedPoints))
                {
                    // The point a almost valid but need to check height
                    if (find_height(height_set, std::get<1>(pq.top()), near_point_threshold) != 0) {
                        double current_height = find_height(height_set, std::get<1>(pq.top()), near_point_threshold);
                        if (abs(current_height - std::get<1>(pq.top())[1]) < 0.15) {
                            Point P_new = std::get<1>(pq.top());
                            //std::cout << "current height" << current_height << endl;
                            //std::cout << "New point location: " << P_new << endl;
                            edge_queue.push(make_pair(P_center, P_new));

                            Sample_Points.push_back(P_new); // Only add point in case A
                            relax_group.push_back(P_new);
                            //std::cout << "Internal angle: " << CGAL::approximate_angle(P_prev, p_slow, P_new) << endl;

                            if (CGAL::orientationC3(p_slow[0], p_fast[0], P_new[0], p_slow[1], p_fast[1], P_new[1], p_slow[2], p_fast[2], P_new[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_new));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_new));
                            area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_new));
                            //std::cout << "Added area: " << sqrt(CGAL::squared_area(p_slow, p_fast, P_new)) << endl;
                            break;
                        }
                        else {
                            pq.pop();
                            continue;
                        }

                    }
                    else { // a new height is found
                        height_set.insert(std::get<1>(pq.top())[1]);
                        Point P_new = std::get<1>(pq.top());
                        //std::cout << "New point location: " << P_new << endl;
                        edge_queue.push(make_pair(P_center, P_new));

                        Sample_Points.push_back(P_new); // Only add point in case A
                        relax_group.push_back(P_new);
                        //std::cout << "Internal angle: " << CGAL::approximate_angle(P_prev, p_slow, P_new) << endl;

                        if (CGAL::orientationC3(p_slow[0], p_fast[0], P_new[0], p_slow[1], p_fast[1], P_new[1], p_slow[2], p_fast[2], P_new[2]) == CGAL::LEFT_TURN)
                            out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_new));
                        else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_new));
                        area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_new));
                        //std::cout << "Added area: " << sqrt(CGAL::squared_area(p_slow, p_fast, P_new)) << endl;
                        break;
                    }

                }

                else pq.pop(); // search pq for the next candidate point
            } // End of pq while loop

            if (pq.empty()) {  // scenario 1
                if (relax_group.size() < 3 && p_slow != p1 && start_point_flag == 3) {
                    if (relax_group.size() == 1) edge_around1.insert(make_pair(relax_group[0], p3));
                    else if (relax_group.size() == 2) {
                        edge_around1.insert(make_pair(relax_group[0], p3));
                        edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                    }
                    edge_queue.push(make_pair(P_center, p1));
                    P_prev = p2;
                    //std::cout << "Initial switch to p1 starting" << endl;

                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));

                            break;
                        }
                    }

                    start_point_flag = 1;
                    edge_queue.pop();
                    continue;
                }
                else if (relax_group.size() == 2 && start_point_flag == 3) { // add the second hexagon near a pentagon, // scenario 2
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 3 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[1], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                            edge_around1.insert(make_pair(p3, relax_group[0]));
                            edge_around2.erase(make_pair(p1, P_exist));
                            edge_around2.erase(make_pair(P_exist, p1));

                            if (CGAL::orientationC3(p1[0], p_fast[0], P_exist[0], p1[1], p_fast[1], P_exist[1], p1[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));

                            //if (CGAL::orientationC3(p1[0], p2[0], P_exist[0], p1[1], p2[1], P_exist[1], p1[2], p2[2], P_exist[2]) == CGAL::LEFT_TURN)
                            //    out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_exist));
                            //else out_mesh.add_face(out_mesh.add_vertex(p2), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));

                            if (CGAL::orientationC3(relax_group[1][0], p_fast[0], P_exist[0], relax_group[1][1], p_fast[1], P_exist[1], relax_group[1][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 2 && start_point_flag == 1) { // add the second hexagon near a pentagon, // scenario 3
                    //std::cout << "scenario 3" << endl;
                    // near boundary, find one new point, switch to p1 starting then find another new point
                    if (sqrt(CGAL::squared_distance(p1, relax_group[1])) < target_distance_p) edge_around1.insert(make_pair(p1, relax_group[1]));

                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {

                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 1 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[1], P_exist));
                            edge_around1.insert(make_pair(p1, relax_group[0]));
                            edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                            edge_around2.erase(make_pair(p3, P_exist));
                            edge_around2.erase(make_pair(P_exist, p3));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));

                            if (CGAL::orientationC3(relax_group[1][0], p_fast[0], P_exist[0], relax_group[1][1], p_fast[1], P_exist[1], relax_group[1][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 1 && P_prev != p1) { // add the second hexagon near a pentagon but only one new point due to a hexagon in place on the other side
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 2 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[0], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_prev));
                            edge_around2.erase(make_pair(p3, P_exist));
                            edge_around2.erase(make_pair(P_exist, p3));
                            edge_around2.erase(make_pair(P_prev, p1));
                            edge_around2.erase(make_pair(p1, P_prev));

                            if (CGAL::orientationC3(p_slow[0], p_fast[0], P_exist[0], p_slow[1], p_fast[1], P_exist[1], p_slow[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_exist));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 1) { // add the second hexagon near a pentagon but only one new point due to a hexagon in place on the other side
                    //std::cout << "scenario 4" << endl;
                    if (start_point_flag == 1) edge_around1.insert(make_pair(relax_group[0], p1)); // only one new point found starting from p1
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 4 Existing point found: " << P_exist << endl;
                            edge_queue.push(make_pair(P_center, P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_prev));
                            edge_around2.erase(make_pair(P_prev, p1));
                            edge_around2.erase(make_pair(p1, P_prev));

                            if (CGAL::orientationC3(relax_group[0][0], p_fast[0], P_exist[0], relax_group[0][1], p_fast[1], P_exist[1], relax_group[0][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[0]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[0]), out_mesh.add_vertex(P_exist));
                        }
                    }
                }

                else if (relax_group.size() == 0 && p_slow != p1 && start_point_flag == 1) {
                    Point P_exist = null_point;
                    start_point_flag = 1;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p2
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p1
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p3
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev
                            && find(flag5_vect.begin(), flag5_vect.end(), CheckNearPoint(p, Sample_Points, near_point_threshold)) == flag5_vect.end()) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            edge_queue.push(make_pair(P_center, P_exist));
                            flag5_vect.push_back(P_exist);
                            //std::cout << "flag 5 Existing point found: " << P_exist << endl;

                            if (CGAL::orientationC3(p_slow[0], p_fast[0], P_exist[0], p_slow[1], p_fast[1], P_exist[1], p_slow[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_exist));

                        }
                    }
                }
                else if (relax_group.size() == 0 && p_slow == p1) { // corner case for generating a hexagon on top of a pentagon but other hexagons in place
                    Point P_exist = null_point;
                    start_point_flag = 1;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p2
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p1) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            edge_queue.push(make_pair(P_center, P_exist));
                            //std::cout << "flag 6 Existing point found: " << P_exist << endl;

                            if (CGAL::orientationC3(p1[0], p_fast[0], P_exist[0], p1[1], p_fast[1], P_exist[1], p1[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));
                        }
                    }

                }
                else {
                    if (CGAL::orientationC3(p_slow[0], P_center[0], p1[0], p_slow[1], P_center[1], p1[1], p_slow[2], P_center[2], p1[2]) == CGAL::LEFT_TURN)
                        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p1));
                    else out_mesh.add_face(out_mesh.add_vertex(P_center), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p1));
                }

            } // end of pq empty if

            else if (relax_group.size() == 3) { // 3 new point and 1 new center point
                edge_around1.insert(make_pair(p3, relax_group[0]));
                edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around1.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around1.insert(make_pair(relax_group[2], p1));
                //std::cout << "3 new point edges around added" << endl;
            }
            //else if (pq.empty() && relax_group.size() < 2) { // boundary
            //    boundary_points.insert(p_slow);
            //    boundary_points.insert(P_center);
            //}

            P_prev = p_slow;
            edge_queue.pop(); // Done with the current edge

        } // End of edge_queue while loop

        //relax_points(relax_group, tree, p1, p3, P_center, r);
        //int n = Sample_Points.size();
        //Sample_Points[n - 3] = relax_group[0];
        //Sample_Points[n - 2] = relax_group[1];
        //Sample_Points[n - 1] = relax_group[2];



        // Keep adding new connected edge pair to the queue
        Connected_edge_queue.pop();
        Connected_edges.clear();
        Connected_edges = findConnectEdge(edge_around1, edge_around2);

        //std::cout << "connected edges size: " << Connected_edges.size() << endl;

        if (Connected_edges.size() == 1 || Connected_edges.size() == 0) { // this hexagon is adjacent to a future pentagon
            queue<pair<Edge, Edge>> copy_queue = Connected_edge_queue;
            while (!copy_queue.empty()) {
                if (p3 == copy_queue.front().first.source || p3 == copy_queue.front().first.target || p3 == copy_queue.front().second.source || p3 == copy_queue.front().second.target) {
                    pentagon_queue.push(Edge(p3, relax_group[0]));
                    edge_around1.erase(make_pair(p3, relax_group[0]));
                    //std::cout << "pentagon edge p3: " << p3 << "   and   " << relax_group[0] << endl;
                    if (Connected_edges.size() == 1) { // check for duplicate of edge pair and pentagon edge
                        if (Connected_edges[0].first.source == relax_group[0] || Connected_edges[0].first.target == relax_group[0] ||
                            Connected_edges[0].second.source == relax_group[0] || Connected_edges[0].second.target == relax_group[0]) {
                            Connected_edges.clear();
                            break;
                        }
                    }
                    break;
                }
                else if (p1 == copy_queue.front().first.source || p1 == copy_queue.front().first.target || p1 == copy_queue.front().second.source || p1 == copy_queue.front().second.target) {
                    if (start_point_flag == 1) {
                        pentagon_queue.push(Edge(p1, relax_group[0]));
                        edge_around1.erase(make_pair(relax_group[0], p1));
                        //std::cout << "pentagon edge p1: " << p1 << "   and   " << relax_group[0] << endl;
                        break;
                    }
                    else {
                        pentagon_queue.push(Edge(p1, relax_group[2]));
                        edge_around1.erase(make_pair(relax_group[2], p1));
                        //std::cout << "pentagon edge p1: " << p1 << "   and   " << relax_group[2] << endl;
                        if (Connected_edges.size() == 1) { // check for duplicate of edge pair and pentagon edge
                            if (Connected_edges[0].first.source == relax_group[2] || Connected_edges[0].first.target == relax_group[2] ||
                                Connected_edges[0].second.source == relax_group[2] || Connected_edges[0].second.target == relax_group[2]) {
                                Connected_edges.clear();
                                break;
                            }
                        }
                        break;
                    }
                }
                else copy_queue.pop();
            }
        }
        edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
        edge_around1.clear();
        for (auto e : Connected_edges) {
            //std::cout << "added edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
            Connected_edge_queue.push(e);
            edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
            edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
        }

        //std::cout << "Connected edge queue size: " << Connected_edge_queue.size() << endl;
        //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;   

    } // end of connected edge_queue
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////



    // Print out uniformity evaluation list

    std::vector<double> distances;

    BOOST_FOREACH(edge_descriptor e, edges(out_mesh))
    {
        halfedge_descriptor hd = halfedge(e, out_mesh);
        double length = CGAL::approximate_sqrt(CGAL::squared_distance(out_mesh.point(target(hd, out_mesh)), out_mesh.point(target(opposite(hd, out_mesh), out_mesh))));

        if (length != 0 && length < target_distance_p) {
            //std::cout << length << endl;
            distances.push_back(length);
        }
    }

    double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    int d_size = distances.size();
    double mean = sum / d_size;
    std::sort(distances.begin(), distances.end());

    std::cout << "Total point-point distances: " << d_size << std::endl;
    std::cout << "The minimum spacing of all point-point distances: " << distances[0] << endl;
    std::cout << "The maximum spacing of all point-point distances: " << distances[d_size - 1] << endl;
    std::cout << "The average spacing of all point-point distances: " << sum / d_size << endl;
    std::cout << "The median spacing of all point-point distances: " << distances[d_size / 2] << endl;
    std::cout << endl;
    evaluate(distances, t);

    std::ofstream out("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/out_mesh.ply");
    CGAL::IO::write_PLY(out, out_mesh);
    out.close();

    //// Empty region related
    //for (Point p1 : boundary_points) {
    //    for (Point p2 : Sample_Points) {
    //        if (abs(p1[0] - p2[0]) < 0.5 && abs(p1[2] - p2[2]) < 0.5 && find(boundary_points.begin(), boundary_points.end(), p2) == boundary_points.end()) {
    //            boundary_points.insert(p2);
    //        }
    //    }
    //}

    //typedef Mesh::Vertex_index Vertex_index;
    //Mesh empty_mesh;
    //Sample_Points.clear();
    //for (auto p : boundary_points) {
    //    if(abs(p[2] - (*boundary_points.begin())[2]) < 5) Sample_Points.push_back(p);
    //}

    //Construct construct(empty_mesh, Sample_Points.begin(), Sample_Points.end());
    //CGAL::advancing_front_surface_reconstruction(Sample_Points.begin(),
    //    Sample_Points.end(),
    //    construct);

    //// Calculate the area of the region
    //double empty_area = mesh_surface_area(empty_mesh);

    //std::cout << "empty region area: " << empty_area << endl;
    //CGAL::IO::write_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/empty_mesh.stl", empty_mesh);

    /*
    // Clear point too close to the border
    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));

    std::set<Point> border_points;
    for (auto b : border) {
        border_points.insert(mesh.point(source(b, mesh)));
        border_points.insert(mesh.point(target(b, mesh)));
    }

    std::vector<Point> final_points;
    for (Point p : Sample_Points) {
        if (check_border(p, border_points, 3)) final_points.push_back(p);
    }
    Sample_Points = final_points;
    */

    int N = Sample_Points.size();
    std::cout << "sample size: " << N << endl;

    /*
    // Writing material B elements output STL file
    Eigen::MatrixXd v1;
    Eigen::MatrixXi f1;
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/1mm_ball.stl", v1, f1);
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    if (N % 2 == 0) {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            ++i;
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }
    else {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            if (i < N - 1) {
                ++i;
                Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
                meshB.push_back(mesh_matrix);
            }
            else {
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp1, f1, f1);
                meshB.push_back(mesh_matrix);
            }
        }
    }


    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;
    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/Uniform__Distribution.stl", get<0>(tmp1), get<1>(tmp1));
    */

    //std::cout << "Surface area: " << area << " and sample area: " << area1 << endl;
    //std::cout << "Empty region ratio of the input mesh: " << (area - area1) * 100 / area << " %" << endl;

    return Sample_Points;

}

vector<Point> create_uniform_points_open(Mesh& mesh, double t, double r, double deviation) {

    vector<Point> Sample_Points;
    vector<Point> relax_group; // usually 4 points
    unordered_set<Point> boundary_points;
    double min_distance = 0.0;
    double target_distance_p = t * (1 + deviation / 100.0);
    double target_distance_n = (t * (1 - deviation / 100.0) > min_distance) ? t * (1 - deviation / 100.0) : min_distance;
    double near_point_threshold = (target_distance_p - target_distance_n);
    double angle_prev = 0.0;
    //cout << "lower bound: " << target_distance_n << " and upper bound: " << target_distance_p << endl;

    Mesh out_mesh; // Sample points mesh
    queue<pair<Point, Point>> edge_queue;
    unordered_set<pair<Point, Point>, pair_hash> edge_around1;
    unordered_set<pair<Point, Point>, pair_hash> edge_around2;
    Point p_slow, p_fast; // p_slow is the starting point

    //if (!PMP::IO::read_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/open_cylinder.stl", mesh))
    //{
    //    std::cerr << "Invalid input." << std::endl;
    //}

    // Clear point too close to the border
    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));

    std::set<vertex_descriptor> border_points;
    for (auto b : border) {
        border_points.insert(source(b, mesh));
        border_points.insert(target(b, mesh));
    }


    double remove_distance = r / 2 * 1.1;


    // Delete points within the distance threshold from the border
    std::set<Point> deletedPoints;
    // Iterate over all points in the mesh
    for (vertex_descriptor v : vertices(mesh))
    {
        // Check the distance from the point to the border points
        for (vertex_descriptor borderV : border_points)
        {
            if (borderV == v) continue;

            if (CGAL::squared_distance(mesh.point(borderV), mesh.point(v)) < remove_distance * remove_distance)
            {
                deletedPoints.insert(mesh.point(v));
                break;
            }
        }
    } // Done with the for loop
    //std::cout << "deletedPoints size: " << deletedPoints.size() << endl;

    // Insert all mesh points in the kd-tree
    Tree tree(mesh.points().begin(), mesh.points().end(), Splitter());

    // Select the starting point (could be random)
    /*
    vertex_iterator vi = vertices(mesh).first;
    int n_advance = 0;
    std::cout << "Enter starting point advance number: ";
    std::cin >> n_advance;
    std::advance(vi, n_advance);

    p_slow = mesh.point(*vi);
    if (!check_border(p_slow, deletedPoints)) {
        std::cout << "Pick another starting point" << endl;
        return 0.0;
    }
    //cout << "First point location: " << mesh.point(*vi)[0] << " " << mesh.point(*vi)[1] << " " << mesh.point(*vi)[2] << endl;
    */


    // Select the starting point from the centroid
    Point p_centroid = CGAL::centroid(mesh.points().begin(), mesh.points().end(), CGAL::Dimension_tag<0>());
    double minDistance = INT32_MAX;
    for (const auto& v : mesh.vertices()) {
        double distance = sqrt(CGAL::squared_distance(mesh.point(v), p_centroid));
        if (distance < minDistance) {
            minDistance = distance;
            p_slow = mesh.point(v);
        }
    }
    std::cout << "start point location: " << p_slow << endl;

    // Done selecting

    std::vector<Point> group1;
    std::vector<Point> group2;

    // Initial search around Point 1 for Point 2
    double area = mesh_surface_area(mesh);
    double area1 = 0;
    //cout << "Surface area: " << area << endl;
    //cout << "vertices number: " << std::distance(mesh.vertices_begin(), mesh.vertices_end()) << endl;
    int K = (PI * r * r) / area * std::distance(mesh.vertices_begin(), mesh.vertices_end()) * 1; // be careful about the multiplier 5
    std::cout << "Defined k number: " << K << endl;
    K_neighbor_search search3(tree, p_slow, K);
    // Find group1
    for (K_neighbor_search::iterator it = search3.begin(); it != search3.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
            group1.push_back(it->first);
        }
    }

    if (group1.size() == 0) {
        std::cout << "Enter a larger multiplier for K number!" << std::endl;
        return Sample_Points;
    }

    // Select the best candidate Point 2 around Point 1
    std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> initial_pq; // distance queue
    for (Point p_it : group1) {
        if (abs(p_it[2] - p_slow[2]) < 0.05) {
            double d = abs(sqrt(CGAL::squared_distance(p_it, p_slow)) - r); // sorted by how close to target distance
            initial_pq.emplace(d, p_it);
        }
    }

    while (!initial_pq.empty()) {
        if (check_border(std::get<1>(initial_pq.top()), deletedPoints)) {
            p_fast = std::get<1>(initial_pq.top()); //  the second point
            break;
        }
        else initial_pq.pop();
    }

    //std::cout << "Initial edge length: " << sqrt(CGAL::squared_distance(p_fast, p_slow)) << endl;
    Sample_Points.push_back(p_slow);
    Sample_Points.push_back(p_fast);
    group1.clear();
    //std::cout << "p_slow: " << p_slow << "     and p_fast: " << p_fast << endl;

    // Find next 4 edges outside the while loop
    K_neighbor_search search2(tree, p_slow, K);
    // Find group1
    for (K_neighbor_search::iterator it = search2.begin(); it != search2.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
            group1.push_back(it->first);
        }
    }

    // Find group2
    K_neighbor_search search1(tree, p_fast, K);
    for (K_neighbor_search::iterator it = search1.begin(); it != search1.end(); it++)
    {
        if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
            group2.push_back(it->first);
        }
    }

    // Select Point 3&4 from the intersection group points
    std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> second_pq; // distance queue
    for (Point it1 : group1) {
        for (Point it2 : group2) {
            if (it1 == it2) {
                double d = pow((sqrt(CGAL::squared_distance(it1, p_slow)) - r), 2) + pow((sqrt(CGAL::squared_distance(it1, p_fast)) - r), 2);
                second_pq.emplace(d, it1);
            }
        }
    }

    // Find point 3
    while (!second_pq.empty()) {
        if (check_border(std::get<1>(second_pq.top()), deletedPoints)) {
            break;
        }
        else second_pq.pop();
    }
    Point P1 = std::get<1>(second_pq.top());
    Point P2;
    Point P_SLOW = p_slow;
    Point P_FAST = p_fast;
    Sample_Points.push_back(P1);
    second_pq.pop();
    // Find point 4
    while (!second_pq.empty()) { // see if a new point can be found
        if (CheckNeighbor(std::get<1>(second_pq.top()), Sample_Points, target_distance_n) && check_border(std::get<1>(second_pq.top()), deletedPoints)) {
            P2 = std::get<1>(second_pq.top());
            break; // point 4 is found
        }
        else second_pq.pop(); // continue searching
    }

    Point P_center = P1;
    edge_queue.push(make_pair(P_center, p_slow));

    // Add faces to out_mesh
    if (CGAL::orientationC3(p_slow[0], p_fast[0], P1[0], p_slow[1], p_fast[1], P1[1], p_slow[2], p_fast[2], P1[2]) == CGAL::LEFT_TURN)
        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P1));
    else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P1));

    if (CGAL::orientationC3(p_slow[0], p_fast[0], P2[0], p_slow[1], p_fast[1], P2[1], p_slow[2], p_fast[2], P2[2]) == CGAL::LEFT_TURN)
        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P2));
    else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P2));

    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P1));
    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P2));
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_fast, P1)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_fast, P2)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_slow, P1)) << endl;
    //std::cout << "New 4 edge length: " << sqrt(CGAL::squared_distance(p_slow, P2)) << endl;
    //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;

    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    Point P_prev = p_fast;
    {
        while (edge_queue.size() > 0) {
            // Clear group search points and update p_slow p_fast by extracting an edge
            group1.clear();
            group2.clear();


            p_slow = edge_queue.front().second;
            p_fast = P_center;

            // Find tow new edges and one candidate point, usually always can find one
            K_neighbor_search search_slow(tree, p_slow, K);
            // Find group1
            for (K_neighbor_search::iterator it = search_slow.begin(); it != search_slow.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
                    group1.push_back(it->first);
                }
            }

            // Find group2
            K_neighbor_search search_fast(tree, p_fast, K);
            for (K_neighbor_search::iterator it = search_fast.begin(); it != search_fast.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
                    group2.push_back(it->first);
                }
            }

            // Get intersection group points
            std::vector<Point> intersection_vect;
            for (Point it1 : group1) {
                for (Point it2 : group2) {
                    if (it1 == it2) {
                        intersection_vect.push_back(it1);
                    }
                }
            }


            std::vector<Point> p_candidate;
            for (Point p : intersection_vect) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) p_candidate.push_back(p);
            }
            //std::cout << "Candidate size: " << p_candidate.size() << endl;

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq; // distance queue
            //std::vector<Point> p_candidate = flag_it == 2 ? p2_vect : p1_vect;
            for (auto p_it : p_candidate) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 2); // sorted by how close to target distance
                pq.emplace(d, p_it);
            }

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_copy = pq;

            while (!pq.empty()) {
                // Can always find a new point in closed surface
                if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && abs(CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) - 120) < 10
                    && check_border(std::get<1>(pq.top()), deletedPoints)) // case A
                {
                    Point P_new = std::get<1>(pq.top()); // Find the valid new point
                    //std::cout << "New formed outside edge length: " << sqrt(CGAL::squared_distance(P_new, p_slow)) << endl;
                    //std::cout << "Internal angle: " << CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) << endl;

                    edge_queue.push(make_pair(P_center, P_new));

                    Sample_Points.push_back(P_new); // Only add point in case A
                    relax_group.push_back(P_new);

                    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_new));
                    //std::cout << "Added area: " << sqrt(CGAL::squared_area(p_slow, p_fast, P_new)) << endl;
                    break;
                }
                else pq.pop(); // search pq for the next candidate point
            } // End of pq while loop

            if (pq.empty()) { //check case B
                // initial 12 points
            }

            edge_queue.pop(); // Done with the current edge
            P_prev = p_slow;

            if (find(Sample_Points.begin(), Sample_Points.end(), P2) == Sample_Points.end() && edge_queue.size() == 0) {

                initial_relax_points(relax_group, tree, P_SLOW, P_FAST, r);
                int n = Sample_Points.size();
                Sample_Points[n - 4] = relax_group[0];
                Sample_Points[n - 3] = relax_group[1];
                Sample_Points[n - 2] = relax_group[2];
                Sample_Points[n - 1] = relax_group[3];
                edge_around1.insert(make_pair(P_SLOW, relax_group[0]));
                edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around1.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around1.insert(make_pair(relax_group[2], relax_group[3]));
                edge_around1.insert(make_pair(relax_group[3], P_FAST));
                relax_group.clear();

                Sample_Points.push_back(P2);
                P_center = P2;
                edge_queue.push(make_pair(P_center, P_SLOW));
                P_prev = P_FAST;

                if (CGAL::orientationC3(Sample_Points[n - 4][0], P_SLOW[0], P1[0], Sample_Points[n - 4][1], P_SLOW[1], P1[1], Sample_Points[n - 4][2], P_SLOW[2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], Sample_Points[n - 3][0], P1[0], Sample_Points[n - 4][1], Sample_Points[n - 3][1], P1[1], Sample_Points[n - 4][2], Sample_Points[n - 3][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 3][0], P1[0], Sample_Points[n - 2][1], Sample_Points[n - 3][1], P1[1], Sample_Points[n - 2][2], Sample_Points[n - 3][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 1][0], P1[0], Sample_Points[n - 2][1], Sample_Points[n - 1][1], P1[1], Sample_Points[n - 2][2], Sample_Points[n - 1][2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P1));

                if (CGAL::orientationC3(Sample_Points[n - 1][0], P_FAST[0], P1[0], Sample_Points[n - 1][1], P_FAST[1], P1[1], Sample_Points[n - 1][2], P_FAST[2], P1[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(P1));
                else out_mesh.add_face(out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P1));
                area1 += sqrt(CGAL::squared_area(P1, P_FAST, Sample_Points[n - 1]));
            }
            else if (edge_queue.size() == 0) {
                initial_relax_points(relax_group, tree, P_SLOW, P_FAST, r);
                int n = Sample_Points.size();
                Sample_Points[n - 4] = relax_group[0];
                Sample_Points[n - 3] = relax_group[1];
                Sample_Points[n - 2] = relax_group[2];
                Sample_Points[n - 1] = relax_group[3];
                edge_around2.insert(make_pair(P_SLOW, relax_group[0]));
                edge_around2.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around2.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around2.insert(make_pair(relax_group[2], relax_group[3]));
                edge_around2.insert(make_pair(relax_group[3], P_FAST));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], P_SLOW[0], P2[0], Sample_Points[n - 4][1], P_SLOW[1], P2[1], Sample_Points[n - 4][2], P_SLOW[2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(P_SLOW), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 4][0], Sample_Points[n - 3][0], P2[0], Sample_Points[n - 4][1], Sample_Points[n - 3][1], P2[1], Sample_Points[n - 4][2], Sample_Points[n - 3][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 4]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 3][0], P2[0], Sample_Points[n - 2][1], Sample_Points[n - 3][1], P2[1], Sample_Points[n - 2][2], Sample_Points[n - 3][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 3]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 2][0], Sample_Points[n - 1][0], P2[0], Sample_Points[n - 2][1], Sample_Points[n - 1][1], P2[1], Sample_Points[n - 2][2], Sample_Points[n - 1][2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(Sample_Points[n - 2]), out_mesh.add_vertex(P2));

                if (CGAL::orientationC3(Sample_Points[n - 1][0], P_FAST[0], P2[0], Sample_Points[n - 1][1], P_FAST[1], P2[1], Sample_Points[n - 1][2], P_FAST[2], P2[2]) == CGAL::LEFT_TURN)
                    out_mesh.add_face(out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(P2));
                else out_mesh.add_face(out_mesh.add_vertex(P_FAST), out_mesh.add_vertex(Sample_Points[n - 1]), out_mesh.add_vertex(P2));
                area1 += sqrt(CGAL::squared_area(P2, P_FAST, Sample_Points[n - 1]));
            }
            //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;
        } // End of edge_queue while loop
    }

    // Generate the first 12 initial points and two cells
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////

    vector<pair<Edge, Edge>> Connected_edges = findConnectEdge(edge_around1, edge_around2); // Edge is a data structure I defined
    queue<Edge> pentagon_queue;
    edge_around2.insert(edge_around1.begin(), edge_around1.end());
    edge_around1.clear();

    queue<pair<Edge, Edge>> Connected_edge_queue;
    for (auto e : Connected_edges) {
        Connected_edge_queue.push(e);
        edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
        edge_around2.erase(make_pair(e.second.source, e.second.target));
    }


    vector<Point> g1, g2, g3;
    vector<Point> flag5_vect;
    int start_point_flag = 3; // 3 means start with p3, otherwise maybe p1
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    while (Connected_edge_queue.size() > 0) {
        g1.clear();
        g2.clear();
        g3.clear();
        relax_group.clear();
        flag5_vect.clear();

        Edge edge1 = Connected_edge_queue.front().first;
        Edge edge2 = Connected_edge_queue.front().second;

        Point p1, p2, p3; // p2 is the middle point
        if (edge1.source == edge2.source || edge1.source == edge2.target) {
            p2 = edge1.source;
            p1 = edge1.target;
            p3 = edge1.source == edge2.source ? edge2.target : edge2.source;
        }
        else {
            p2 = edge1.target;
            p1 = edge1.source;
            p3 = edge1.target == edge2.source ? edge2.target : edge2.source;
        }
        //std::cout << "p1 location: " << p1 << "     p2 location: " << p2 << "     p3 location: " << p3 << endl;
        //std::cout << "p123 angle: " << CGAL::approximate_angle(p1, p2, p3) << endl;
        if (sqrt(CGAL::squared_distance(p1, p2)) > target_distance_p || sqrt(CGAL::squared_distance(p2, p3)) > target_distance_p
            || sqrt(CGAL::squared_distance(p1, p2)) < target_distance_n || sqrt(CGAL::squared_distance(p2, p3)) < target_distance_n) { // invalid edge pair
            Connected_edge_queue.pop();
            continue;
        }
        edge_around2.erase(make_pair(p1, p2));
        edge_around2.erase(make_pair(p2, p1));
        edge_around2.erase(make_pair(p3, p2));
        edge_around2.erase(make_pair(p2, p3));

        // Find a new center point to generate a cell around
        {
            // p1
            K_neighbor_search search_p1(tree, p1, K);
            for (K_neighbor_search::iterator it = search_p1.begin(); it != search_p1.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p1) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p1) >= pow(target_distance_n, 2)) {
                    g1.push_back(it->first);
                }
            }

            // p2
            K_neighbor_search search_p2(tree, p3, K);
            for (K_neighbor_search::iterator it = search_p2.begin(); it != search_p2.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p3) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p3) >= pow(target_distance_n, 2)) {
                    g2.push_back(it->first);
                }
            }
            // p3
            K_neighbor_search search_p3(tree, p3, K);
            for (K_neighbor_search::iterator it = search_p3.begin(); it != search_p3.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p3) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it->first, p3) >= pow(target_distance_n, 2)) {
                    g3.push_back(it->first);
                }
            }

            std::vector<Point> center_vect;
            std::vector<Point> center_vect_final;
            std::sort(g1.begin(), g1.end());
            std::sort(g2.begin(), g2.end());
            std::sort(g3.begin(), g3.end());
            std::set_intersection(g1.begin(), g1.end(), g2.begin(), g2.end(), std::back_inserter(center_vect));
            std::set_intersection(center_vect.begin(), center_vect.end(), g3.begin(), g3.end(), std::back_inserter(center_vect_final));

            std::vector<Point> pq_vect;
            for (Point p : center_vect_final) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) pq_vect.push_back(p);
            }
            //std::cout << "Center point candidate size: " << pq_vect.size() << endl;

            if (pq_vect.size() == 0) // need to generate a pentagon
            {
                if (center_vect_final.size() > 0) { // empty region when growing front meets
                    Point P_exist = null_point;
                    for (Point p : center_vect_final) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "boundary existing point found" << endl;
                            break;
                        }
                    }
                    if (P_exist == null_point) {
                        boundary_points.insert(p1);
                        boundary_points.insert(p2);
                        boundary_points.insert(p3);
                        //std::cout << "boundary point due to no new center" << endl;
                    }
                }

                //std::cout << "No new center point, pentagon case" << endl;
                queue<Edge> copy_queue = pentagon_queue;
                Point P_penta1, P_penta2; // each from a hexagon cell, used to find the new pentagon point
                while (!copy_queue.empty()) {

                    if (copy_queue.front().source == p1 || copy_queue.front().source == p3)
                    {
                        if (sqrt(CGAL::squared_distance(copy_queue.front().source, copy_queue.front().target)) < target_distance_p) {
                            P_penta1 = copy_queue.front().target;
                            P_penta2 = copy_queue.front().source == p1 ? p3 : p1;
                            //std::cout << "Penta point: " << P_penta1 << "   " << P_penta2 << endl;
                            break;
                        }
                        else copy_queue.pop();
                    }
                    else if (copy_queue.front().target == p1 || copy_queue.front().target == p3) {
                        if (sqrt(CGAL::squared_distance(copy_queue.front().source, copy_queue.front().target)) < target_distance_p) {
                            P_penta1 = copy_queue.front().source;
                            P_penta2 = copy_queue.front().source == p1 ? p3 : p1;
                            //std::cout << "Penta point: " << P_penta1 << "   " << P_penta2 << endl;
                            break;
                        }
                        else copy_queue.pop();
                    }
                    else copy_queue.pop();
                }

                K_neighbor_search search_penta1(tree, P_penta1, K);
                K_neighbor_search search_penta2(tree, P_penta2, K);
                std::vector<Point> penta_intersect;
                for (K_neighbor_search::iterator it1 = search_penta1.begin(); it1 != search_penta1.end(); it1++)
                {
                    if (CGAL::squared_distance(it1->first, P_penta1) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it1->first, P_penta1) >= pow(target_distance_n, 2)) {
                        for (K_neighbor_search::iterator it2 = search_penta2.begin(); it2 != search_penta2.end(); it2++)
                        {
                            if (CGAL::squared_distance(it2->first, P_penta2) <= pow(target_distance_p * 1.05, 2) && CGAL::squared_distance(it2->first, P_penta2) >= pow(target_distance_n, 2))
                            {
                                if (it1->first == it2->first && CheckNeighbor(it1->first, Sample_Points, target_distance_n)) penta_intersect.push_back(it1->first);
                            }
                            else continue;
                        }
                    }
                    else continue;
                }

                if (penta_intersect.size() == 0) {
                    //std::cout << "No pentagon intersection points" << endl << endl;
                    edge_around1.insert(make_pair(P_penta2, P_penta1));
                    Connected_edges.clear();
                    Connected_edges = findConnectEdge(edge_around1, edge_around2);
                    edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
                    edge_around1.clear();

                    for (auto e : Connected_edges) {
                        //std::cout << "added pentagon edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
                        Connected_edge_queue.push(e);
                        edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
                        edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
                    }
                    Connected_edge_queue.pop();
                    continue; // go to the next connected edge pair
                }

                std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_penta;
                for (auto p_it : penta_intersect) {
                    double d = pow(sqrt(CGAL::squared_distance(p_it, P_penta1)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, P_penta2)) - r, 2);
                    pq_penta.emplace(d, p_it);
                }

                Point P_new = std::get<1>(pq_penta.top()); // insert a pentagon point
                Sample_Points.push_back(P_new);
                //std::cout << "New pentagon point: " << P_new << endl << endl;
                edge_around1.insert(make_pair(P_new, P_penta1));
                edge_around1.insert(make_pair(P_new, P_penta2));
                Connected_edges.clear();
                Connected_edges = findConnectEdge(edge_around1, edge_around2);
                edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
                edge_around1.clear();

                for (auto e : Connected_edges) {
                    //std::cout << "added pentagon edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
                    Connected_edge_queue.push(e);
                    edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
                    edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
                }

                Connected_edge_queue.pop();
                continue; // go to the next connected edge pair
            } // end of dealing with pentagon case

            // Now a new center point is found
            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_center; // distance queue
            for (auto p_it : pq_vect) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p1)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p2)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p3)) - r, 2);
                //double d = pow(sqrt(CGAL::squared_distance(p_it, p2)) - r, 2);
                pq_center.emplace(d, p_it);
            }

            if (CheckNeighbor(std::get<1>(pq_center.top()), Sample_Points, target_distance_n) && check_border(std::get<1>(pq_center.top()), deletedPoints)) {
                P_center = std::get<1>(pq_center.top());
                Sample_Points.push_back(P_center);
                //std::cout << "New center point: " << P_center << endl;
            }
            else {
                Connected_edge_queue.pop();
                continue;
            }
        }

        // Add two faces and three edges defined by connected two edges
        edge_queue.push(make_pair(P_center, p3));
        start_point_flag = 3;

        if (CGAL::orientationC3(p1[0], p2[0], P_center[0], p1[1], p2[1], P_center[1], p1[2], p2[2], P_center[2]) == CGAL::LEFT_TURN)
            out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_center));
        else out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p2));
        area1 += sqrt(CGAL::squared_area(p1, p2, P_center));

        if (CGAL::orientationC3(p3[0], p2[0], P_center[0], p3[1], p2[1], P_center[1], p3[2], p2[2], P_center[2]) == CGAL::LEFT_TURN)
            out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_center));
        else out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p2));
        area1 += sqrt(CGAL::squared_area(p3, p2, P_center));

        P_prev = p2;

        // Now generate a new cell around P_center
        while (edge_queue.size() > 0) {
            group1.clear();
            group2.clear();
            //pq_copy.swap(pq_empty); // clear the pq_copy by swapping with an empty queue

            p_slow = edge_queue.front().second;
            p_fast = P_center;


            // Find tow new edges and one candidate point, usually always can find one
            K_neighbor_search search_slow(tree, p_slow, K);
            // Find group1
            for (K_neighbor_search::iterator it = search_slow.begin(); it != search_slow.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_slow) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_slow) >= pow(target_distance_n, 2)) {
                    group1.push_back(it->first);
                }
            }

            // Find group2
            K_neighbor_search search_fast(tree, p_fast, K);
            for (K_neighbor_search::iterator it = search_fast.begin(); it != search_fast.end(); it++)
            {
                if (CGAL::squared_distance(it->first, p_fast) <= pow(target_distance_p, 2) && CGAL::squared_distance(it->first, p_fast) >= pow(target_distance_n, 2)) {
                    group2.push_back(it->first);
                }
            }

            // Get intersection group points
            std::vector<Point> intersection_vect;
            for (Point it1 : group1) {
                for (Point it2 : group2) {
                    if (it1 == it2) {
                        intersection_vect.push_back(it1);
                    }
                }
            }

            std::vector<Point> p_candidate;
            for (Point p : intersection_vect) {
                if (CheckNeighbor(p, Sample_Points, target_distance_n)) p_candidate.push_back(p);
            }

            //if (p_candidate.empty() && !intersection_vect.empty()) { // either there is an existing point or front boundary or no points at all
            //    Point P_exist = null_point;
            //    for (Point p : intersection_vect) {
            //        if (CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n) != null_point
            //            && sqrt(CGAL::squared_distance(p, P_prev)) > target_distance_n
            //            && find(relax_group.begin(), relax_group.end(), CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n)) == relax_group.end()
            //            && relax_group.size() < 2) { // case C

            //            if (start_point_flag == 1 && sqrt(CGAL::squared_distance(p, p1)) > target_distance_p) {
            //                P_exist = CheckBoundaryPoint(p, Sample_Points, near_point_threshold, target_distance_n);
            //                std::cout << "insert boundary point as existing point found: " << P_exist << endl;
            //                std::cout << "inserted: " << p_slow << endl;
            //                boundary_points.insert(p_slow);
            //                break;
            //            }
            //            else if (start_point_flag == 3 && relax_group.size() == 1) {
            //                std::cout << "inserted: " << p_slow << endl;
            //                boundary_points.insert(p_slow);
            //            }

            //        }
            //    }
            //}

            std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq; // distance queue
            //std::vector<Point> p_candidate = flag_it == 2 ? p2_vect : p1_vect;
            for (auto p_it : p_candidate) {
                double d = pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 4) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 4); // sorted by how close to target distance
                //double d = 1.0/r/r/2.0*(pow(sqrt(CGAL::squared_distance(p_it, p_slow)) - r, 2) + pow(sqrt(CGAL::squared_distance(p_it, p_fast)) - r, 2)) + 1.0/120.0*abs(CGAL::approximate_angle(P_prev, p_slow, p_it) - 120);
                pq.emplace(d, p_it);
            }

            //std::priority_queue<std::tuple<double, Point>, std::vector<std::tuple<double, Point>>, std::greater<std::tuple<double, Point>>> pq_copy = pq; // copy the queue for later use

            while (!pq.empty()) {
                // Can always find a new point in closed surface
                // if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && abs(CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) - 120) < 15) // case A
                if (CheckNeighbor(std::get<1>(pq.top()), Sample_Points, target_distance_n) && CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) < 123
                    && CGAL::approximate_angle(P_prev, p_slow, std::get<1>(pq.top())) > 117 && check_border(std::get<1>(pq.top()), deletedPoints))
                {
                    Point P_new = std::get<1>(pq.top());
                    //std::cout << "New point location: " << P_new << endl;
                    edge_queue.push(make_pair(P_center, P_new));

                    Sample_Points.push_back(P_new); // Only add point in case A
                    relax_group.push_back(P_new);

                    if (CGAL::orientationC3(p_slow[0], p_fast[0], P_new[0], p_slow[1], p_fast[1], P_new[1], p_slow[2], p_fast[2], P_new[2]) == CGAL::LEFT_TURN)
                        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_new));
                    else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_new));
                    area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_new));
                    //std::cout << "Added area: " << sqrt(CGAL::squared_area(p_slow, p_fast, P_new)) << endl;
                    break;
                }

                else pq.pop(); // search pq for the next candidate point
            } // End of pq while loop

            if (pq.empty()) {  // scenario 1
                if (relax_group.size() < 3 && p_slow != p1 && start_point_flag == 3) {
                    if (relax_group.size() == 1) edge_around1.insert(make_pair(relax_group[0], p3));
                    else if (relax_group.size() == 2) {
                        edge_around1.insert(make_pair(relax_group[0], p3));
                        edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                    }
                    edge_queue.push(make_pair(P_center, p1));
                    P_prev = p2;
                    //std::cout << "Initial switch to p1 starting" << endl;

                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(p3, p_fast, P_exist));

                            break;
                        }
                    }

                    start_point_flag = 1;
                    edge_queue.pop();
                    continue;
                }
                else if (relax_group.size() == 2 && start_point_flag == 3) { // add the second hexagon near a pentagon, // scenario 2
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 3 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[1], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                            edge_around1.insert(make_pair(p3, relax_group[0]));
                            edge_around2.erase(make_pair(p1, P_exist));
                            edge_around2.erase(make_pair(P_exist, p1));

                            if (CGAL::orientationC3(p1[0], p_fast[0], P_exist[0], p1[1], p_fast[1], P_exist[1], p1[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));

                            //if (CGAL::orientationC3(p1[0], p2[0], P_exist[0], p1[1], p2[1], P_exist[1], p1[2], p2[2], P_exist[2]) == CGAL::LEFT_TURN)
                            //    out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p2), out_mesh.add_vertex(P_exist));
                            //else out_mesh.add_face(out_mesh.add_vertex(p2), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));

                            if (CGAL::orientationC3(relax_group[1][0], p_fast[0], P_exist[0], relax_group[1][1], p_fast[1], P_exist[1], relax_group[1][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 2 && start_point_flag == 1) { // add the second hexagon near a pentagon, // scenario 3
                    //std::cout << "scenario 3" << endl;
                    // near boundary, find one new point, switch to p1 starting then find another new point
                    if (sqrt(CGAL::squared_distance(p1, relax_group[1])) < target_distance_p) edge_around1.insert(make_pair(p1, relax_group[1]));

                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {

                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 1 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[1], P_exist));
                            edge_around1.insert(make_pair(p1, relax_group[0]));
                            edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                            edge_around2.erase(make_pair(p3, P_exist));
                            edge_around2.erase(make_pair(P_exist, p3));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(p3, p_fast, P_exist));

                            if (CGAL::orientationC3(relax_group[1][0], p_fast[0], P_exist[0], relax_group[1][1], p_fast[1], P_exist[1], relax_group[1][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[1]), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(relax_group[1], p_fast, P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 1 && P_prev != p1) { // add the second hexagon near a pentagon but only one new point due to a hexagon in place on the other side
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 2 Existing point found: " << P_exist << endl;
                            edge_around1.insert(make_pair(relax_group[0], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_prev));
                            edge_around2.erase(make_pair(p3, P_exist));
                            edge_around2.erase(make_pair(P_exist, p3));
                            edge_around2.erase(make_pair(P_prev, p1));
                            edge_around2.erase(make_pair(p1, P_prev));

                            if (CGAL::orientationC3(p_slow[0], p_fast[0], P_exist[0], p_slow[1], p_fast[1], P_exist[1], p_slow[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_exist));

                            if (CGAL::orientationC3(p3[0], p_fast[0], P_exist[0], p3[1], p_fast[1], P_exist[1], p3[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p3), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p3), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(p3, p_fast, P_exist));

                            break;
                        }
                    }
                }
                else if (relax_group.size() == 1) { // add the second hexagon near a pentagon but only one new point due to a hexagon in place on the other side
                    //std::cout << "scenario 4" << endl;
                    if (start_point_flag == 1) edge_around1.insert(make_pair(relax_group[0], p1)); // only one new point found starting from p1
                    Point P_exist = null_point;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            //std::cout << "flag 4 Existing point found: " << P_exist << endl;
                            edge_queue.push(make_pair(P_center, P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_exist));
                            edge_around1.insert(make_pair(relax_group[0], P_prev));
                            edge_around2.erase(make_pair(P_prev, p1));
                            edge_around2.erase(make_pair(p1, P_prev));

                            if (CGAL::orientationC3(relax_group[0][0], p_fast[0], P_exist[0], relax_group[0][1], p_fast[1], P_exist[1], relax_group[0][2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(relax_group[0]), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(relax_group[0]), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(relax_group[0], p_fast, P_exist));
                        }
                    }
                }

                else if (relax_group.size() == 0 && p_slow != p1 && start_point_flag == 1) {
                    Point P_exist = null_point;
                    start_point_flag = 1;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p2
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p1
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p3
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != P_prev
                            && find(flag5_vect.begin(), flag5_vect.end(), CheckNearPoint(p, Sample_Points, near_point_threshold)) == flag5_vect.end()) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            edge_queue.push(make_pair(P_center, P_exist));
                            flag5_vect.push_back(P_exist);
                            //std::cout << "flag 5 Existing point found: " << P_exist << endl;

                            if (CGAL::orientationC3(p_slow[0], p_fast[0], P_exist[0], p_slow[1], p_fast[1], P_exist[1], p_slow[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_exist));
                            area1 += sqrt(CGAL::squared_area(p_slow, p_fast, P_exist));

                        }
                    }
                }
                else if (relax_group.size() == 0 && p_slow == p1) { // corner case for generating a hexagon on top of a pentagon but other hexagons in place
                    Point P_exist = null_point;
                    start_point_flag = 1;
                    for (Point p : intersection_vect) {
                        if (CheckNearPoint(p, Sample_Points, (near_point_threshold)) != null_point
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p2
                            && CheckNearPoint(p, Sample_Points, (near_point_threshold)) != p1) {
                            P_exist = CheckNearPoint(p, Sample_Points, (near_point_threshold));
                            edge_queue.push(make_pair(P_center, P_exist));
                            //std::cout << "flag 6 Existing point found: " << P_exist << endl;

                            if (CGAL::orientationC3(p1[0], p_fast[0], P_exist[0], p1[1], p_fast[1], P_exist[1], p1[2], p_fast[2], P_exist[2]) == CGAL::LEFT_TURN)
                                out_mesh.add_face(out_mesh.add_vertex(p1), out_mesh.add_vertex(p_fast), out_mesh.add_vertex(P_exist));
                            else out_mesh.add_face(out_mesh.add_vertex(p_fast), out_mesh.add_vertex(p1), out_mesh.add_vertex(P_exist));
                        }
                    }

                }
                else {
                    if (CGAL::orientationC3(p_slow[0], P_center[0], p1[0], p_slow[1], P_center[1], p1[1], p_slow[2], P_center[2], p1[2]) == CGAL::LEFT_TURN)
                        out_mesh.add_face(out_mesh.add_vertex(p_slow), out_mesh.add_vertex(P_center), out_mesh.add_vertex(p1));
                    else out_mesh.add_face(out_mesh.add_vertex(P_center), out_mesh.add_vertex(p_slow), out_mesh.add_vertex(p1));
                    area1 += sqrt(CGAL::squared_area(p1, p_slow, P_center));
                }

            } // end of pq empty if

            else if (relax_group.size() == 3) { // 3 new point and 1 new center point
                edge_around1.insert(make_pair(p3, relax_group[0]));
                edge_around1.insert(make_pair(relax_group[0], relax_group[1]));
                edge_around1.insert(make_pair(relax_group[1], relax_group[2]));
                edge_around1.insert(make_pair(relax_group[2], p1));
                //std::cout << "3 new point edges around added" << endl;
            }
            //else if (pq.empty() && relax_group.size() < 2) { // boundary
            //    boundary_points.insert(p_slow);
            //    boundary_points.insert(P_center);
            //}

            P_prev = p_slow;
            edge_queue.pop(); // Done with the current edge

        } // End of edge_queue while loop

        //relax_points(relax_group, tree, p1, p3, P_center, r);
        //int n = Sample_Points.size();
        //Sample_Points[n - 3] = relax_group[0];
        //Sample_Points[n - 2] = relax_group[1];
        //Sample_Points[n - 1] = relax_group[2];



        // Keep adding new connected edge pair to the queue
        Connected_edge_queue.pop();
        Connected_edges.clear();
        Connected_edges = findConnectEdge(edge_around1, edge_around2);

        //std::cout << "connected edges size: " << Connected_edges.size() << endl;

        if (Connected_edges.size() == 1 || Connected_edges.size() == 0) { // this hexagon is adjacent to a future pentagon
            queue<pair<Edge, Edge>> copy_queue = Connected_edge_queue;
            while (!copy_queue.empty()) {
                if (p3 == copy_queue.front().first.source || p3 == copy_queue.front().first.target || p3 == copy_queue.front().second.source || p3 == copy_queue.front().second.target) {
                    pentagon_queue.push(Edge(p3, relax_group[0]));
                    edge_around1.erase(make_pair(p3, relax_group[0]));
                    //std::cout << "pentagon edge p3: " << p3 << "   and   " << relax_group[0] << endl;
                    if (Connected_edges.size() == 1) { // check for duplicate of edge pair and pentagon edge
                        if (Connected_edges[0].first.source == relax_group[0] || Connected_edges[0].first.target == relax_group[0] ||
                            Connected_edges[0].second.source == relax_group[0] || Connected_edges[0].second.target == relax_group[0]) {
                            Connected_edges.clear();
                            break;
                        }
                    }
                    break;
                }
                else if (p1 == copy_queue.front().first.source || p1 == copy_queue.front().first.target || p1 == copy_queue.front().second.source || p1 == copy_queue.front().second.target) {
                    if (start_point_flag == 1) {
                        pentagon_queue.push(Edge(p1, relax_group[0]));
                        edge_around1.erase(make_pair(relax_group[0], p1));
                        //std::cout << "pentagon edge p1: " << p1 << "   and   " << relax_group[0] << endl;
                        break;
                    }
                    else {
                        pentagon_queue.push(Edge(p1, relax_group[2]));
                        edge_around1.erase(make_pair(relax_group[2], p1));
                        //std::cout << "pentagon edge p1: " << p1 << "   and   " << relax_group[2] << endl;
                        if (Connected_edges.size() == 1) { // check for duplicate of edge pair and pentagon edge
                            if (Connected_edges[0].first.source == relax_group[2] || Connected_edges[0].first.target == relax_group[2] ||
                                Connected_edges[0].second.source == relax_group[2] || Connected_edges[0].second.target == relax_group[2]) {
                                Connected_edges.clear();
                                break;
                            }
                        }
                        break;
                    }
                }
                else copy_queue.pop();
            }
        }
        edge_around2.insert(edge_around1.begin(), edge_around1.end()); // merge two sets
        edge_around1.clear();
        for (auto e : Connected_edges) {
            //std::cout << "added edge pair: " << e.first.source << "   " << e.first.target << "   and   " << e.second.source << "   " << e.second.target << endl;
            Connected_edge_queue.push(e);
            edge_around2.erase(make_pair(e.first.source, e.first.target)); // remove used connected edge pair
            edge_around2.erase(make_pair(e.second.source, e.second.target)); // remove used connected edge pair
        }

        //std::cout << "Connected edge queue size: " << Connected_edge_queue.size() << endl;
        //std::cout << "New sample size: " << Sample_Points.size() << endl << endl;

    } // end of connected edge_queue
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////



    // Print out uniformity evaluation list

    std::vector<double> distances;

    BOOST_FOREACH(edge_descriptor e, edges(out_mesh))
    {
        halfedge_descriptor hd = halfedge(e, out_mesh);
        double length = CGAL::approximate_sqrt(CGAL::squared_distance(out_mesh.point(target(hd, out_mesh)), out_mesh.point(target(opposite(hd, out_mesh), out_mesh))));

        if (length != 0 && length < target_distance_p) {
            //std::cout << length << endl;
            distances.push_back(length);
        }
    }

    double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    int d_size = distances.size();
    double mean = sum / d_size;
    std::sort(distances.begin(), distances.end());

    std::cout << "Total point-point distances: " << d_size << std::endl;
    std::cout << "The minimum spacing of all point-point distances: " << distances[0] << endl;
    std::cout << "The maximum spacing of all point-point distances: " << distances[d_size - 1] << endl;
    std::cout << "The average spacing of all point-point distances: " << sum / d_size << endl;
    std::cout << "The median spacing of all point-point distances: " << distances[d_size / 2] << endl;
    std::cout << endl;
    evaluate(distances, r);

    std::ofstream out("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/out_mesh.ply");
    CGAL::IO::write_PLY(out, out_mesh);
    out.close();

    int N = Sample_Points.size();
    std::cout << "sample size: " << N << endl;

    /*
    // Writing material B elements output STL file
    Eigen::MatrixXd v1;
    Eigen::MatrixXi f1;
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/1mm_ball.stl", v1, f1);
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    if (N % 2 == 0) {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            ++i;
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }
    else {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            if (i < N - 1) {
                ++i;
                Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
                meshB.push_back(mesh_matrix);
            }
            else {
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp1, f1, f1);
                meshB.push_back(mesh_matrix);
            }
        }
    }


    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;
    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/Uniform__patch.stl", get<0>(tmp1), get<1>(tmp1));
    */

    //std::cout << "Surface area: " << area << " and sample area: " << area1 << endl;
    //std::cout << "Empty region ratio of the input mesh: " << (area - area1) * 100 / area << " %" << endl;

    return Sample_Points;

}


// For surface patching
typedef K::Compare_dihedral_angle_3                    Compare_dihedral_angle_3;
template <typename G>
struct Constraint : public boost::put_get_helper<bool, Constraint<G> >
{
    typedef boost::readable_property_map_tag      category;
    typedef bool                                  value_type;
    typedef bool                                  reference;
    typedef edge_descriptor                       key_type;
    Constraint()
        :g_(NULL)
    {}
    Constraint(G& g, double bound)
        : g_(&g), bound_(bound)
    {}
    bool operator[](edge_descriptor e) const
    {
        const G& g = *g_;
        return compare_(g.point(source(e, g)),
            g.point(target(e, g)),
            g.point(target(next(halfedge(e, g), g), g)),
            g.point(target(next(opposite(halfedge(e, g), g), g), g)),
            bound_) == CGAL::SMALLER;
    }
    const G* g_;
    Compare_dihedral_angle_3 compare_;
    double bound_;
};
template <typename PM>
struct Put_true
{
    Put_true(const PM pm)
        :pm(pm)
    {}
    template <typename T>
    void operator()(const T& t)
    {
        put(pm, t, true);
    }
    PM pm;
};

void surface_patching() {
    Mesh mesh; // Input mesh
    //double t = 0.0;
    //std::cout << "Please enter the target distance: ";
    //std::cin >> t;
    double r = 0.0;
    std::cout << "Please enter the adjusted target distance: ";
    std::cin >> r;
    double deviation = 0.0;
    std::cout << "Please enter the tolerance (percentage): ";
    std::cin >> deviation;

    if (!PMP::IO::read_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/open_cylinder.stl", mesh))
    {
        std::cerr << "Invalid input." << std::endl;
    }

    //const double bound = std::cos(0.75 * CGAL_PI);

    //// Create a property map to store the component index for each face
    //Mesh::Property_map<face_descriptor, std::size_t> fccmap =
    //    mesh.add_property_map<face_descriptor, std::size_t>("f:CC").first;

    //std::size_t num = PMP::connected_components(mesh,
    //    fccmap,
    //    PMP::parameters::edge_is_constrained_map(Constraint<Mesh>(mesh, bound)));

    //std::cerr << "- The graph has " << num << " connected components (face connectivity)" << std::endl;

    //// Store the connected components as separate meshes
    //std::vector<Mesh> connected_components(num+1);

    //// Iterate over the faces of the input mesh
    //for (face_descriptor f : faces(mesh))
    //{
    //    std::size_t cc = fccmap[f]; // Get the component index for the face
    //    //std::cout << "at patch: " << cc << endl;

    //    // Retrieve the halfedge descriptor for the face
    //    halfedge_descriptor he = halfedge(f, mesh);

    //    // Retrieve the vertex descriptors for the face by iterating over the incident halfedges
    //    vertex_descriptor v0 = source(he, mesh);
    //    vertex_descriptor v1 = target(he, mesh);
    //    vertex_descriptor v2 = target(next(he, mesh), mesh);

    //    Mesh::Vertex_index iv1 = connected_components[cc].add_vertex(mesh.point(v0));
    //    Mesh::Vertex_index iv2 = connected_components[cc].add_vertex(mesh.point(v1));
    //    Mesh::Vertex_index iv3 = connected_components[cc].add_vertex(mesh.point(v2));

    //    //// Add the vertices and the face to the corresponding connected component mesh
    //    connected_components[cc].add_face(iv1, iv2, iv3);
    //}

    //CGAL::IO::write_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/patch_mesh1.stl", connected_components[1]);

    //Mesh patch_mesh;
    //if (!PMP::IO::read_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/patch_mesh1.stl", patch_mesh))
    //{
    //    std::cerr << "Invalid input." << std::endl;
    //}

    //std::vector<Point> Sample_Points = create_uniform_points_open(patch_mesh, 3, r, deviation);
    std::vector<Point> Sample_Points = create_uniform_points(mesh, 5, r, deviation);
    //std::vector<Point> Sample_Points_merge;
    //for (Point p : Sample_Points1) {
    //    if (pow(p[0], 2) + pow(p[2], 2) < 23 * 23) continue; // special case points screening - surface with a hole
    //    else Sample_Points_merge.push_back(p);
    //}
    //for (Point p : Sample_Points2) {
    //    Sample_Points_merge.push_back(p);
    //}

    //std::vector<Point> Sample_Points;
    ////std::cout << "checking intersection" << endl;
    //for (Point p1 : Sample_Points_merge) {
    //    for (Point p2 : Sample_Points_merge) {
    //        if (p1 != p2 && sqrt(CGAL::squared_distance(p1, p2)) > 4 && find(Sample_Points.begin(), Sample_Points.end(), p1) == Sample_Points.end()) {
    //            Sample_Points.push_back(p1);
    //        }
    //    }
    //}
    std::cout << "Final Sample size: " << Sample_Points.size() << endl;
    // Writing material B elements output STL file
    Eigen::MatrixXd v1;
    Eigen::MatrixXi f1;
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/1mm_ball.stl", v1, f1);
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    int N = Sample_Points.size();
    if (N % 2 == 0) {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            ++i;
            Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
    }
    else {
        for (int i = 0; i < N; ++i) {
            Eigen::MatrixXd v_tmp1 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
            if (i < N - 1) {
                ++i;
                Eigen::MatrixXd v_tmp2 = move_mesh(v1, Sample_Points[i][0], Sample_Points[i][1], Sample_Points[i][2]);
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
                meshB.push_back(mesh_matrix);
            }
            else {
                std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp1, f1, f1);
                meshB.push_back(mesh_matrix);
            }
        }
    }


    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;
    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/Uniform__patch.stl", get<0>(tmp1), get<1>(tmp1));

}



void embed_materialB() {
    Mesh mesh; // Input mesh

    if (!PMP::IO::read_polygon_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/open_cylinder.stl", mesh))
    {
        std::cerr << "Invalid input." << std::endl;
    }
    std::vector<Point> samples_pos = create_uniform_points(mesh, 5, 4.88, 10);
    std::cout << "Done sampling" << endl;

    std::map<Point, vertex_descriptor> points;

    for (const auto& vd : mesh.vertices())
    {
        const Point& vertexPoint = mesh.point(vd);
        points[vertexPoint] = vd;
    }

    std::map<vertex_descriptor, Vector_3> vnormals;
    PMP::compute_vertex_normals(mesh, boost::make_assoc_property_map(vnormals), PMP::parameters::vertex_point_map(mesh.points()).geom_traits(K()));

    std::vector<Vector_3> samples_nor;
    for (Point p : samples_pos) {
        vertex_descriptor vid = points[p];
        samples_nor.push_back(vnormals[vid]);
    }

    // Now generate material B based on grid points
    Eigen::MatrixXd v1;
    Eigen::MatrixXi f1;
    int N = samples_pos.size();
    printf("Generating %i meshB elements... \n", N);
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/zero_cylinder.stl", v1, f1);
    vector<std::tuple<MatrixXd, MatrixXi>> meshB;

    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d Nvec1(samples_nor[i].x(), samples_nor[i].y(), samples_nor[i].z());
        Eigen::MatrixXd Rv1 = rotate_mesh(v1, Nvec1);
        //Eigen::MatrixXd v_tmp1 = move_mesh(v1, samples_pos[i].x, samples_pos[i].y, samples_pos[i].z);
        Eigen::MatrixXd v_tmp1 = move_mesh(Rv1, samples_pos[i].x(), samples_pos[i].y(), samples_pos[i].z());

        if (i < N - 1) {
            ++i;
            Eigen::Vector3d Nvec2(samples_nor[i].x(), samples_nor[i].y(), samples_nor[i].z());
            Eigen::MatrixXd Rv2 = rotate_mesh(v1, Nvec2);
            //Eigen::MatrixXd v_tmp2 = move_mesh(v1, samples_pos[i].x, samples_pos[i].y, samples_pos[i].z);
            Eigen::MatrixXd v_tmp2 = move_mesh(Rv2, samples_pos[i].x(), samples_pos[i].y(), samples_pos[i].z());

            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp2, f1, f1);
            meshB.push_back(mesh_matrix);
        }
        else {
            std::tuple<MatrixXd, MatrixXi> mesh_matrix = connect_meshes(v_tmp1, v_tmp1, f1, f1);
            meshB.push_back(mesh_matrix);
        }

    }

    std::tuple<MatrixXd, MatrixXi> tmp1 = meshB[0];
    std::tuple<MatrixXd, MatrixXi> tmp2;
    // Generate material B based on saved locations
    for (int k = 1; k < meshB.size(); ++k) {
        tmp2 = connect_meshes(get<0>(tmp1), get<0>(meshB[k]), get<1>(tmp1), get<1>(meshB[k]));
        tmp1 = tmp2;
    }
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB__Distribution.stl", get<0>(tmp1), get<1>(tmp1));


}



int main(int argc, char* argv[])
{


    //cout << "Material B Generation." << endl;
    //cout << "Please enter the target triangular MESH SIZE for all opeartions (default 5.0): ";
    //cin >> TARGET_EDGE_LENGTH;
    //cout << endl;


    //std::tuple<MatrixXd, MatrixXi> materialB_mesh = generate_meshB();
    //cout << "Finished generating material B!" << endl;
    //main_v2 = get<0>(materialB_mesh);
    //main_f2 = get<1>(materialB_mesh);
    //igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB_TEST_small.stl", main_v2, main_f2);


    /*
    igl::read_triangle_mesh("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/materialB/meshB_TEST_small.stl", main_v2, main_f2);

    string filename = "C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Application/Part4.STL";
    CGAL_remesh(filename);
    igl::read_triangle_mesh(filename, main_v1, main_f1);

     //Visualize the relative position of material A & B
    MatrixXd main_V(main_v1.rows() + main_v2.rows(), main_v1.cols());
    main_V << main_v1, main_v2;
    MatrixXi main_F(main_f1.rows() + main_f2.rows(), main_f1.cols());
    main_F << main_f1, (main_f2.array() + main_v1.rows());

    std::cout << "Use the number keyboard under function keys to shift material A" << endl;
    std::cout << "RED - X axis; GREEN - Y axis; BLUE - Z axis" << endl;
    std::cout <<
        "Press 1 to move material A along X axis by 10" << endl <<
        "Press 2 to move material A along X axis by -10" << endl <<
        "Press 3 to move material A along Y axis by 10" << endl <<
        "Press 4 to move material A along Y axis by -10" << endl <<
        "Press 5 to move material A along Z axis by 10" << endl <<
        "Press 6 to move material A along Z axis by -10" << endl;
    std::cout << "Close the viewer window after shifting of material A is finished" << endl << endl;

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(main_V, main_F);;
    Eigen::MatrixXd Axis_P1(3, 3);
    Axis_P1 <<
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;
    Eigen::MatrixXd Axis_P2(3, 3);
    Axis_P2 <<
        800, 0, 0,
        0, 800, 0,
        0, 0, 800;

    viewer.data().add_edges(Axis_P1.row(0), Axis_P2.row(0), Eigen::RowVector3d(1, 0, 0));
    viewer.data().add_edges(Axis_P1.row(1), Axis_P2.row(1), Eigen::RowVector3d(0, 1, 0));
    viewer.data().add_edges(Axis_P1.row(2), Axis_P2.row(2), Eigen::RowVector3d(0, 0, 1));
    viewer.callback_key_down = &key_down;
    viewer.launch();

    std::cout << "Now boolean ..." << endl;
    igl::copyleft::cgal::mesh_boolean(main_v1, main_f1, main_v2, main_f2, igl::MESH_BOOLEAN_TYPE_MINUS, main_v3, main_f3);
    igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Application/testBOOLEAN.STL", main_v3, main_f3);
    */

    //igl::copyleft::cgal::mesh_boolean(v1, f1, v2, f2, igl::MESH_BOOLEAN_TYPE_INTERSECT, v4, f4);
    //igl::writeSTL("C:/Users/Derek Zhang/Desktop/WAAM Research/CAD/Application/testBOOLEAN_MaterialB.STL", v4, f4);


    //create_poisson_points();
    //evaluate();

    //std::vector<int> vect;
    //std::vector<double> result;
    //for (int i = 15; i <= 60; i++) {
    //    vect.push_back(i);
    //}

    //for (auto i : vect) {
    //    std::cout << "Current r = " << i << endl;
    //    double percentage = create_uniform_points(i, 8);
    //    result.push_back(percentage);
    //}

    //for (auto j : result) {
    //    std::cout << j << endl;
    //}

    //double t = 0.0;
    //std::cout << "Please enter the target distance: ";
    //std::cin >> t;
    //double r = 0.0;
    //std::cout << "Please enter the adjusted target distance: ";
    //std::cin >> r;
    //double deviation = 0.0; 
    //std::cout << "Please enter the tolerance (percentage): ";
    //std::cin >> deviation;
    //int n_stop = 0;
    //cout << "Enter stop size: ";
    //cin >> n_stop;

    //create_uniform_points_open(r, deviation, n_stop);
    //create_uniform_points(5, 4.85, 10, 1000);
    surface_patching();
    //embed_materialB();


    std::cout << "Finished !" << endl;
    return EXIT_SUCCESS;
}
