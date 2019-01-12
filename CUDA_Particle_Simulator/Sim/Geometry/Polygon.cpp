#include "Polygon.h"

Polygon::Polygon(){};

Polygon::Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
{
      v[0] = v1;
      v[1] = v2;
      v[2] = v3;
}


bool Polygon::checkPolygonIntersection(glm::vec3 x_0, glm::vec3 x_1, CollisionData &result)
{

    // http://geomalgorithms.com/a06-_intersect-2.html
    result = CollisionData();
    glm::vec3 x_01 = (x_1 - x_0);

    glm::vec3 v_0  = v[0];
    glm::vec3 v_1  = v[1];
    glm::vec3 v_2  = v[2];

    glm::vec3 v_01 = v_1 - v_0;
    glm::vec3 v_02 = v_2 - v_0;

    glm::vec3 n = glm::normalize( glm::cross( ( v_1 - v_0 ), ( v_2 - v_0 ) ) );
    float r_i = ( glm::dot(n, v_0 - x_0) ) / ( glm::dot(n, x_1 - x_0) );
    if (!(0 <= r_i && r_i <= 1))
    {
	    return false;
    }

    glm::vec3 i(x_0 + r_i * x_01);

    // check if intersection lies within triangle
    float uu, uv, vv, wu, wv, d;
    uu = glm::dot(v_01, v_01);
    uv = glm::dot(v_01, v_02);
    vv = glm::dot(v_02, v_02);
    glm::vec3 w = i - v_0;
    wu = glm::dot(w,v_01);
    wv = glm::dot(w,v_02);
    d = uv * uv - uu * vv;

    //test parametric co-ords
    float s, t;
    s = (uv * wv - vv * wu) / d;
    t = (uv * wu - uu * wv) / d;

    if (s >= 0 && t >= 0 && s+t <= 1) {
		    //std::cout << "collision!!!! particle " << p_indx << " -> polygon " << j << std::endl;
		    std::cout << "collision!!!" << std::endl;
		    result = CollisionData(i, n, r_i); 
		    return true;
    }
    
    return false; 

}

Polygon::Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 _n)
{
      v[0] = v1;
      v[1] = v2;
      v[2] = v3;
	n = glm::normalize(_n);
}

void Polygon::print() const
{
      std::cout << (float)v[0].x << ", "<< (float)v[0].y << ", "<< (float)v[0].z << " | "
	          << (float)v[1].x << ", "<< (float)v[1].y << ", "<< (float)v[1].z << " | "
		    << (float)v[2].x << ", "<< (float)v[2].y << ", "<< (float)v[2].z << " | " << std::endl;
}
