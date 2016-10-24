//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tamasi Benjamin
// Neptun : A7TWOS
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)

#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>

#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif

#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple

#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char *log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete[] log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, const char *message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330 core
    precision highp float;

	uniform mat4 MVP;

	layout (location = 0) in vec3 position;
	layout (location = 1) in vec3 color;
	out vec3 ourColor;

	void main() {
		ourColor = color;
		gl_Position = vec4(position, 1.0f) * MVP;
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330 core
    precision highp float;

	in vec3 ourColor;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(ourColor, 1.0f); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {}

    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    mat4 operator*(const mat4 &right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }

    operator float *() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];

    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v[3] = w;
    }

    vec4 operator*(const mat4 &mat) const{
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }

    vec4 operator%(const vec4  &vec) const {
        return vec4(
                y() * vec.z() - (z() * vec.y()),
                z() * vec.x() - (x() * vec.z()),
                x() * vec.y() - (y() * vec.x())
        );
    }

    vec4 operator-(const vec4 &vec) const {
        return vec4(x()-vec.x(),y()-vec.y(),z()-vec.z());
    }

    float length() const {
        return sqrtf(x()*x() + y()*y() + z()*z());
    }

    vec4 normal() const {
        float len = length();
        return vec4(x()/len, y()/len, z()/len);
    }

    float x() const {return v[0]/v[3];}
    float y() const {return v[1]/v[3];}
    float z() const {return v[2]/v[3];}
};

// 3D camera
struct Camera {
    vec4 c; // center
    vec4 t; // target
    float fov; // field of view
    float far, near;
public:
    Camera() {
        c = vec4(0, 0, 10);
        t = vec4();
        fov = 45.0f;
        far = 100.0f;
        near = 0.1f;
        Animate(0);
    }

    mat4 V() {
        vec4 U(0.0f, 1.0f, 0.0f);
        vec4 D = (t - c).normal();
        vec4 R((U%D).normal());

        mat4 A(
                R.x(), U.x(), D.x(), 0.0f,
                R.y(), U.y(), D.y(), 0.0f,
                R.z(), U.z(), D.z(), 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
        );

        return Vold() * A;
    }

    mat4 Vold() { // view matrix: translates the center to the origin
        return mat4(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    -c.x(), -c.y(), -c.z(), 1);
    }

    mat4 P() { // projection matrix (perspective)
        float Q = far / (far - near);
        return mat4(1 / tanf(fov / 2), 0, 0, 0,
                    0, 1 / tanf(fov / 2), 0, 0,
                    0, 0, Q, 1,
                    0, 0, -Q * near, 1);
    }

    mat4 Vinv() { // inverse view matrix
        return mat4(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    c.x(), c.y(), c.z(), 1);
    }

    mat4 Pinv() { // inverse projection matrix
        float Q = far / (far - near);
        return mat4(1 / tanf(fov / 2), 0, 0, 0,
                    0, 1 / tanf(fov / 2), 0, 0,
                    0, 0, Q, 1,
                    0, 0, -Q * near, 1);
    }

    void Animate(float t) {
        c.v[0] = 2.5f * sinf(t);
        c.v[1] = 0 ;//sinf(t);
        c.v[2] = 2.5f * cosf(t);// + 0.5f * cosf(t);
    }
};

// 3D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

struct ColoredVertex {
public:
    vec4 v;
    vec4 c;

    ColoredVertex(vec4 vertex, vec4 color) : v(vertex), c(color) {}
};

struct Drawable {
    GLuint vao, vbo;

protected:
    std::vector<ColoredVertex> vertices;
    GLenum draw_type;

public:
    float sx, sy, sz;        // scaling
    float wTx, wTy, wTz;     // translation
    Drawable() {
        sx = sy = sz = 1;
        wTx = wTy = wTz = 0;
        Animate(0);
    }

    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid *) 0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid *) (3 * sizeof(float)));

        float *vertexData = new float[vertices.size() * 6];
        for (int i = 0; i < vertices.size(); i++) {
            vertexData[i * 6 + 0] = vertices[i].v.v[0]; // x
            vertexData[i * 6 + 1] = vertices[i].v.v[1]; // y
            vertexData[i * 6 + 2] = vertices[i].v.v[2]; // z
            vertexData[i * 6 + 3] = vertices[i].c.v[0]; // R
            vertexData[i * 6 + 4] = vertices[i].c.v[1]; // G
            vertexData[i * 6 + 5] = vertices[i].c.v[2]; // B
        }
        // copy data to the GPU
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * 6 * sizeof(float), vertexData, GL_STATIC_DRAW);
        delete[] vertexData;
    }

    virtual void Animate(float t) {
    }

    void Draw() {
        mat4 M(sx, 0, 0, 0,
               0, sy, 0, 0,
               0, 0, sz, 0,
               wTx, wTy, wTz, 1); // model matrix

        mat4 MVPTransform = M * camera.V() * camera.P();

        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
        else printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        glDrawArrays(draw_type, 0, (GLsizei) vertices.size());    // draw vertices
    }
};

struct Triangle : public Drawable {
public:
    Triangle(ColoredVertex a, ColoredVertex b, ColoredVertex c) {
        // Model coords
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);

        draw_type = GL_TRIANGLES;
    }
};

struct Pyramid : public Drawable {
public:
    Pyramid() {
        ColoredVertex a = ColoredVertex(vec4(-1, -1, -1), vec4(1));
        ColoredVertex b = ColoredVertex(vec4(1, -1, -1), vec4(0, 1));
        ColoredVertex c = ColoredVertex(vec4(0, -1, 1), vec4(0, 0, 1));
        ColoredVertex d = ColoredVertex(vec4(0, 1, 0), vec4());

        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);

        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(d);

        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);

        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(d);

        draw_type = GL_TRIANGLES;
    }
};

// The virtual world
Triangle t1(
        ColoredVertex(vec4(-1, -1), vec4(0, 0, 1)),
        ColoredVertex(vec4(0, 1), vec4(1, 0, 0)),
        ColoredVertex(vec4(1, -1), vec4(0, 1, 0.5))
);

Pyramid p1;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    // Create objects by setting up their vertex data on the GPU
    //t1.Create();
    p1.Create();

    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");

    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");

    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Connect Attrib Arrays to input variables of the vertex shader
    glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
    glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");    // fragmentColor goes to the frame buffer memory

    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);

    glEnable(GL_DEPTH_TEST);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

    p1.Draw();
    //t1.Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case 'q':
            exit(0);
        default:
            break;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON &&
        state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        //lineStrip.AddPoint(cX, cY);
        glutPostRedisplay();     // redraw
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;                // convert msec to sec
    camera.Animate(sec);                    // animate the camera

    glutPostRedisplay();                    // redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth,
                       windowHeight);                // Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);                            // Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH |
                        GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;    // magic
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}