from helper_classes import *
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)

            intersection_point, nearest_object = find_intersection(ray, objects)
            if intersection_point is not None:
                intersection_point += get_point_bias(nearest_object, intersection_point)
                # We calculate the color of the pixel using the Phong model.
                color = get_color(nearest_object, ambient, intersection_point,
                                  nearest_object.compute_normal(intersection_point),
                                  lights, camera, objects, 1, max_depth, ray)
                # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)
    return image


def get_color(obj, ambient, intersection_point, normal, lights, view_point, objects, recursion_depth, max_depth, ray):
    # Using the Phong model to calculate the color of the pixel
    Ia = obj.ambient * ambient
    sum_Id_Is = np.zeros(3)

    for light in lights:
        intersection_light_intensity = light.get_intensity(intersection_point)
        light_direction = light.get_light_ray(intersection_point).direction

        # View vector
        view_direction = normalize(view_point - intersection_point)

        # Reflection vector
        reflected_ray = reflected(-light_direction, normal)

        if shadow_coefficient(light, intersection_point, objects):

            # Diffuse component
            Id = obj.diffuse * intersection_light_intensity * np.dot(normal, light_direction)

            # Specular component
            Is = obj.specular * intersection_light_intensity * \
                (np.dot(view_direction, reflected_ray) ** obj.shininess)

            sum_Id_Is += Id + Is
    if recursion_depth < max_depth:
        reflected_ray = Ray(intersection_point, reflected(ray.direction, obj.compute_normal(intersection_point)))
        intersection_point, nearest_object = find_intersection(reflected_ray, objects)
        if nearest_object is not None:
            intersection_point += get_point_bias(nearest_object, intersection_point)
            normal = nearest_object.compute_normal(intersection_point)
            reflected_color = get_color(nearest_object, ambient, intersection_point,
                                        normal, lights, view_point, objects, recursion_depth + 1, max_depth, reflected_ray)
            sum_Id_Is += obj.reflection * reflected_color
    color = Ia + sum_Id_Is
    return color


def find_intersection(ray: Ray, objects):
    nearest_object, min_distance = Ray.nearest_intersected_object(ray, objects)
    if nearest_object is None:
        return None, None

    intersection_point = ray.origin + (min_distance * ray.direction)
    return intersection_point, nearest_object


def shadow_coefficient(light, point, objects):
    light_ray = light.get_light_ray(point)
    nearest_object, min_distance = light_ray.nearest_intersected_object(objects)
    if nearest_object is None:
        return True

    distance = light.get_distance_from_light(point)
    if min_distance < distance:
        return False

    return True

def get_point_bias(object, point):
    return 0.01 * object.compute_normal(point)

# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
