import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    reflected_vector = vector - 2 * np.dot(vector, normal) * normal
    return reflected_vector


## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection_point):
        ray = Ray(intersection_point, -self.direction)
        return ray

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        f_att = 1 / (self.kc + self.kl * d + self.kq * (d**2))
        return self.intensity * f_att


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        f_att = 1 / (self.kc + self.kl * d + self.kq * (d**2))
        source_to_point = normalize(self.get_light_ray(intersection).direction)
        Il = self.intensity * np.dot(source_to_point, -self.direction) * f_att

        return Il



class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf

        for obj in objects:
            intersection = obj.intersect(self)
            if intersection is not None:
                distance, _ = intersection
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = obj

        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        # v is the vector from the origin of the ray to the point on the plane
        v = self.point - ray.origin
        distance = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)

        if distance > 0:
            return distance, self
        else:
            return None

    def compute_normal(self, _):
        return self.normal


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        self.edge1 = self.b - self.a
        self.edge2 = self.c - self.a
        self.normal = self.compute_normal(a)

    def intersect(self, ray: Ray):
        triangle_plane = Plane(self.normal, self.c)
        intersection = triangle_plane.intersect(ray)
        if intersection is None:
            return None

        distance, _ = intersection
        intersection_point = ray.origin + (distance * ray.direction)
        if self.barycentric_intersection(intersection_point):
            return intersection
        return None

    def barycentric_intersection(self, intersection_point):
        p = intersection_point
        pa = p - self.a
        pb = p - self.b
        pc = p - self.c

        # Calculate the area of the triangle
        area_ABC = np.linalg.norm(np.cross(self.edge1, self.edge2))
        area_PBC = np.linalg.norm(np.cross(pb, pc))
        area_PCA = np.linalg.norm(np.cross(pc, pa))
        area_PAB = np.linalg.norm(np.cross(pa, pb))

        # Barycentric coordinates
        alpha = area_PBC / area_ABC
        beta = area_PCA / area_ABC
        gamma = area_PAB / area_ABC

        # Check if point is inside the triangle
        return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and np.abs(alpha + beta + gamma - 1) < 1e-8

    # computes normal to the triangle surface. Pay attention to its direction!
    def compute_normal(self, _):
        return normalize(np.cross(self.edge1, self.edge2))


class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()
        self.last_intersected_triangle = None

    def create_triangle_list(self):
        triangle_list = []
        t_idx = [
                [0, 1, 3],
                [1, 2, 3],
                [0, 3, 2],
                [4, 1, 0],
                [4, 2, 1],
                [2, 4, 0]]

        for triangle_idx in t_idx:
            a = self.v_list[triangle_idx[0]]
            b = self.v_list[triangle_idx[1]]
            c = self.v_list[triangle_idx[2]]
            triangle_list.append(Triangle(a, b, c))

        return triangle_list

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        closest_intersection = None

        for triangle in self.triangle_list:
            result = triangle.intersect(ray)
            if result is not None:
                if closest_intersection is None or result[0] < closest_intersection[0]:
                    closest_intersection = result
                    self.last_intersected_triangle = triangle

        return closest_intersection

    def compute_normal(self, _):
        return self.last_intersected_triangle.compute_normal(_)



class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        origin_to_center_vector = ray.origin - self.center
        b_coefficient = 2 * np.dot(ray.direction, origin_to_center_vector)
        c_coefficient = np.dot(origin_to_center_vector, origin_to_center_vector) - self.radius ** 2
        discriminant = b_coefficient ** 2 - 4 * c_coefficient

        if discriminant < 0:
            return None
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            distance_1 = (-b_coefficient - sqrt_discriminant) / 2
            distance_2 = (-b_coefficient + sqrt_discriminant) / 2

            if distance_1 > 0 and distance_2 > 0:
                return min(distance_1, distance_2), self
            else:
                distance = max(distance_1, distance_2)
                if distance > 0:
                    return distance, self
                else:
                    return None

    def compute_normal(self, intersection_point):
        return normalize(intersection_point - self.center)
