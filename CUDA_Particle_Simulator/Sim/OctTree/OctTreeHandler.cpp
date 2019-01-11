#include "OctTreeHandler.h"

OctTreeHandler::OctTreeHandler()
{
      _doRender = false;
}

OctTreeHandler::~OctTreeHandler() {}

void OctTreeHandler::buildTree(std::vector<Polygon> p, Volume vol)
{
      std::cout << "Building Tree" << std::endl;
      OctTree root(0);
      root.vol = vol;

      node_list.push_back(&root);
      root.generateOctTree(p, vol, 0, 0, node_list);

      std::cout << "Number Nodes : " << node_list.size() << std::endl;
}

void OctTreeHandler::toggleRender()
{
      if (_doRender) _doRender = false;
      else _doRender = true;
}

void OctTreeHandler::clear()
{
      node_list.clear();
}

void OctTreeHandler::generateOpenGLData()
{
      _positions.clear();
      glm::vec3 A,B,C,D,E,F,G,H;
      glm::vec4 col(0.0f, 1.0f, 0.0f, 1.0f);

      for (OctTree* o: node_list)
      {
            float w;
            A = o->vol.BBL;
            G = o->vol.TTR;

            w = G[0] - A[0];

            B = glm::vec3(A[0]+w, A[1], A[2]);
            C = glm::vec3(A[0]+w, A[1]+w, A[2]);
            D = glm::vec3(A[0], A[1]+w, A[2]);

            F = glm::vec3(G[0], G[1]-w, G[2]);
            H = glm::vec3(G[0]-w, G[1], G[2]);
            E = glm::vec3(G[0]-w, G[1]-w, G[2]);

            _positions.push_back(A); _positions.push_back(B);
            _positions.push_back(B); _positions.push_back(C);
            _positions.push_back(C); _positions.push_back(D);
            _positions.push_back(D); _positions.push_back(A);

            _positions.push_back(E); _positions.push_back(F);
            _positions.push_back(F); _positions.push_back(G);
            _positions.push_back(G); _positions.push_back(H);
            _positions.push_back(H); _positions.push_back(E);

            _positions.push_back(A); _positions.push_back(E);
            _positions.push_back(B); _positions.push_back(F);
            _positions.push_back(C); _positions.push_back(G);
            _positions.push_back(D); _positions.push_back(H);

            for (int i = 0; i < 24; i++) _colors.push_back(col);

      }
}


bool OctTreeHandler::init(GLuint* programs)
{

      glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);

      generateOpenGLData();

      _model_matrix = mat4(1.0);

      glGenBuffers(2, _buffers); //Create two buffer objects, one for vertex positions and one for vertex colors

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _positions.size() * sizeof(vec3), _positions.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[1]);  //Buffers[1] will be the color for each vertex
	glBufferData(GL_ARRAY_BUFFER, _colors.size() * sizeof(vec4), _colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(1);

	_pvm_matrix_loc = glGetUniformLocation(programs[0], "_pvm_matrix");

	_initialized = true;

	return true;
}


void OctTreeHandler::draw(GLuint* programs, mat4 proj_mat, mat4 view_mat)
{
      if (_doRender)
      {
            mat4 _pvm_matrix = proj_mat * view_mat * _model_matrix;
      	glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));

      	glUseProgram(programs[0]);
      	if (!_initialized)
      	{
      		std::cout << "ERROR : Cannot  render an object thats not initialized. Plane\n";
      		return;
      	}
      	glBindVertexArray(_vao);
      	glLineWidth(0.5f);
      	glDrawArrays(GL_LINES, 0, _positions.size());
      }
}
