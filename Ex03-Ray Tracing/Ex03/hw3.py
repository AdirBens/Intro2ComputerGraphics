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
                normal = nearest_object.compute_normal(intersection_point)
                intersection_point += get_point_bias(nearest_object, intersection_point)
                # We calculate the color of the pixel using the Phong model.
                color = get_color(nearest_object, ambient, intersection_point, normal, lights, camera,
                                  objects, max_depth, ray, recursion_depth=1)

           # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)
    return image


def get_color(obj, ambient, intersection_point, normal, lights, view_point, objects, max_depth, ray, recursion_depth):
    # Using the Phong model to calculate the color of the pixel
    Ia = obj.ambient * ambient
    sum_Id_Is = np.zeros(3)

    for light in lights:
        intersection_light_intensity = light.get_intensity(intersection_point)
        light_direction = light.get_light_ray(intersection_point).direction

        # View vector
        view_vector = normalize(view_point - intersection_point)

        # Reflection vector
        reflected_ray = reflected(-light_direction, normal)

        if shadow_coefficient(light, intersection_point, objects):

            # Diffuse component
            Id = obj.diffuse * intersection_light_intensity * np.dot(normal, light_direction)

            # Specular component
            Is = obj.specular * intersection_light_intensity * \
                (np.dot(view_vector, reflected_ray) ** obj.shininess)

            sum_Id_Is += Id + Is
    if recursion_depth < max_depth:
        reflected_ray = Ray(intersection_point, reflected(ray.direction, obj.compute_normal(intersection_point)))
        intersection_point, nearest_object = find_intersection(reflected_ray, objects)
        if nearest_object is not None:
            normal = nearest_object.compute_normal(intersection_point)
            intersection_point += get_point_bias(nearest_object, intersection_point)
            reflected_color = get_color(nearest_object, ambient, intersection_point,
                                        normal, lights, view_point, objects, max_depth,
                                        reflected_ray, recursion_depth + 1)
            sum_Id_Is += obj.reflection * reflected_color

    color = Ia + sum_Id_Is
    return color


def find_intersection(ray: Ray, objects):
    """
    Finds the intersection of the ray with the objects in the scene.
    """
    nearest_object, min_distance = Ray.nearest_intersected_object(ray, objects)
    if nearest_object is None:
        return None, None

    intersection_point = ray.origin + (min_distance * ray.direction)
    return intersection_point, nearest_object


def shadow_coefficient(light, point, objects):
    """
    Calculates the shadow coefficient of the light source.
    """
    light_ray = light.get_light_ray(point)
    nearest_object, min_distance = light_ray.nearest_intersected_object(objects)
    if nearest_object is None:
        return True

    distance = light.get_distance_from_light(point)
    if min_distance < distance:
        return False

    return True


def get_point_bias(object, point):
    """
    Returns a small bias to the intersection.
    """
    return 0.01 * object.compute_normal(point)


def your_own_scene():
    camera = np.array([0, 0, 1])
    objects = []
    lights = []

    ground_plane = Plane([0, 1, 0], [0, -1, 0])
    ground_plane.set_material(
        ambient=[0.2, 0.2, 0.4],
        diffuse=[0.5, 0.5, 0.7],
        specular=[0.5, 0.5, 0.7],
        shininess=2000,
        reflection=0
    )
    objects.append(ground_plane)

    sphere1 = Sphere([-1.5, 1,-1.5],1.6)
    sphere1.set_material(np.array([0, 1, 0]), [1, 0, 0], [0, 0, 1], 2000, 1)
    objects.append(sphere1)

    sphere2 = Sphere([1.5, 1, -1.5], 0.5)
    sphere2.set_material(np.array([0.8, 0.3, 0.2]), [0.5, 0.1, 0.8], [0, 0, 1], 100, 0.5)
    objects.append(sphere2)

    v_list = np.array(
    [
        [0.5, 0, 0],
        [-0.5, 0, 0],
        [0, 0.5, 0],
        [0, -0.8, 0],
        [0, 0, 0.5]
    ])

    diamond = Pyramid(v_list)
    diamond.set_material(
        ambient=[0.8, 0.2, 0.2],
        diffuse=[0.2, 0.5, 0.5],
        specular=[0.2, 0.5, 0.5],
        shininess=10,
        reflection=0.2
    )
    objects.append(diamond)

    light1 = PointLight(
        intensity=np.array([0.3, 0.3, 0.3]),
        position=np.array([2, 1, 1]),
        kc=0.1,
        kl=0.1,
        kq=0.1
    )
    lights.append(light1)

    light2 = SpotLight(
        intensity=np.array([0.7, 0.7, 0.7]),
        position=np.array([1, 0, 0]),
        direction=np.array([-1, 0, 0]),
        kc=0.1,
        kl=0.1,
        kq=0.1
    )
    lights.append(light2)

    return camera, lights, objects
