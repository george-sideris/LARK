import adsk.core, adsk.fusion, traceback
import math
import os

def sample_line(startPoint, endPoint, increment):
    """Sample points along a line segment."""
    length = startPoint.distanceTo(endPoint) * 10
    num_points = max(int(length / increment) + 1, 2)  # Ensure at least two points
    points = []
    for i in range(num_points):
        t = i * increment / length
        x = startPoint.x + t * (endPoint.x - startPoint.x)
        y = startPoint.y + t * (endPoint.y - startPoint.y)
        z = startPoint.z + t * (endPoint.z - startPoint.z)
        points.append(adsk.core.Point3D.create(x, y, z))
    return points

def sample_arc(startPoint, endPoint, centerPoint, radius, increment):
    """Sample points along an arc segment."""
    startAngle = math.atan2(startPoint.y - centerPoint.y, startPoint.x - centerPoint.x)
    endAngle = math.atan2(endPoint.y - centerPoint.y, endPoint.x - centerPoint.x)
    if startAngle > endAngle:
        startAngle, endAngle = endAngle, startAngle
    arcLength = radius * (endAngle - startAngle) * 10
    num_points = max(int(arcLength / increment) + 1, 2)  # Ensure at least two points
    points = []
    for i in range(num_points):
        angle = startAngle + i * (endAngle - startAngle) / (num_points - 1)
        x = centerPoint.x + radius * math.cos(angle)
        y = centerPoint.y + radius * math.sin(angle)
        z = startPoint.z  # Assuming the arc lies on the same plane
        points.append(adsk.core.Point3D.create(x, y, z))
    return points

def transform_points(points, transform, scale_factor):
    """Apply a transformation matrix and scale to a list of 3D points."""
    transformed_points = []
    for point in points:
        point.transformBy(transform)
        transformed_points.append(adsk.core.Point3D.create(
            point.x * scale_factor, 
            point.y * scale_factor, 
            point.z * scale_factor
        ))
    return transformed_points

def get_next_file_number(directory, base_name):
    """Find the next available file number."""
    max_num = -1
    for file_name in os.listdir(directory):
        if file_name.startswith(base_name) and file_name.endswith('.txt'):
            try:
                num = int(file_name[len(base_name):-4])  # Extract the number from the file name
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    return max_num + 1

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent

        # Get the selected entity
        selection = ui.activeSelections
        if selection.count == 0:
            # ui.messageBox('No object selected. Please select an edge or curve.')
            return
        
        selectedObject = selection.item(0).entity

        if not isinstance(selectedObject, adsk.fusion.BRepEdge) and not isinstance(selectedObject, adsk.fusion.SketchCurve):
            ui.messageBox('Selected object is not a supported curve type.')
            return

        # Define the sampling increment
        increment = 0.05  # Distance between the construction planes

        # Create an empty list to store the sampled points
        points = []

        if isinstance(selectedObject, adsk.fusion.BRepEdge):
            # Handle BRepEdge (for 3D edges)
            edge = selectedObject
            edge_length = edge.length * 100

            # Create construction planes and intersect with sketch
            planes = rootComp.constructionPlanes
            num_planes = int(edge_length / increment) + 1
            for i in range(num_planes):
                try:
                    # Create plane at a specific distance along the edge
                    distance = adsk.core.ValueInput.createByReal(i * increment)
                    planeInput = planes.createInput()
                    planeInput.setByDistanceOnPath(edge, distance)
                    plane = planes.add(planeInput)

                    # Create a sketch on the plane
                    sketch = rootComp.sketches.add(plane)

                    # Intersect the sketch plane with the edge
                    skPoints = sketch.intersectWithSketchPlane([edge])
                    sketchTransform = sketch.transform
                    for skPoint in skPoints:
                        point = skPoint.geometry
                        point.transformBy(sketchTransform)
                        points.append(point)
                except Exception as e:
                    print(f"Error at plane {i}: {e}")
                    continue
        
        elif isinstance(selectedObject, adsk.fusion.SketchCurve):
            # Handle SketchCurve (for sketch curves)
            sketch = selectedObject.parentSketch
            if isinstance(selectedObject, adsk.fusion.SketchLine):
                startPoint = selectedObject.startSketchPoint.geometry
                endPoint = selectedObject.endSketchPoint.geometry
                points.extend(sample_line(startPoint, endPoint, increment))
            elif isinstance(selectedObject, adsk.fusion.SketchArc):
                startPoint = selectedObject.startSketchPoint.geometry
                endPoint = selectedObject.endSketchPoint.geometry
                centerPoint = selectedObject.centerSketchPoint.geometry
                radius = startPoint.distanceTo(centerPoint)
                points.extend(sample_arc(startPoint, endPoint, centerPoint, radius, increment))
            elif isinstance(selectedObject, adsk.fusion.SketchSpline):
                fitPoints = selectedObject.fitPoints
                num_points = len(fitPoints)
                if num_points < 2:
                    ui.messageBox('Not enough fit points to sample spline.')
                    return
                for i in range(num_points - 1):
                    p1 = fitPoints[i].geometry
                    p2 = fitPoints[i + 1].geometry
                    points.extend(sample_line(p1, p2, increment))
            else:
                ui.messageBox('Unsupported sketch curve type.')
                return

        if not points:
            ui.messageBox('No points sampled from the selected curve.')
            return

        # Get the occurrence transformation
        occurrence = selectedObject.assemblyContext
        if occurrence:
            transform = occurrence.transform
        else:
            transform = adsk.core.Matrix3D.create()  # Identity matrix if no occurrence

        # Transform points to model space and scale to millimeters
        scale_factor = 10.0  # Assuming the unit is centimeters, convert to millimeters
        transformed_points = transform_points(points, transform, scale_factor)

        # Determine the next file name
        directory = 'C:/Users/georg/Documents/Landmark Trajectories/'
        base_name = 'b'
        next_number = get_next_file_number(directory, base_name)
        file_path = os.path.join(directory, f'{base_name}{next_number}.txt')

        # Write the points to the file
        with open(file_path, 'w') as f:
            for point in transformed_points:
                f.write(f'{point.x}, {point.y}, {point.z}\n')

        ui.messageBox(f'Points exported successfully to {file_path}.')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

# Run the script
run(None)
