import adsk.core, adsk.fusion, adsk.cam, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent

        # Get the selected entities
        selection = ui.activeSelections
        if selection.count == 0:
            ui.messageBox('No points selected. Please select construction points.')
            return
        
        points = []
        for selectionItem in selection:
            selectedEntity = selectionItem.entity
            if isinstance(selectedEntity, adsk.fusion.ConstructionPoint):
                point = selectedEntity
                points.append(point)
            else:
                ui.messageBox('Selected entity is not a ConstructionPoint.')
                return
        
        if not points:
            ui.messageBox('No construction points selected.')
            return

        # Extract the coordinates
        coordinates = [(point.geometry.x * 10, point.geometry.y * 10, point.geometry.z * 10) for point in points]

        # Write the coordinates to a file
        file_path = 'C:/Users/georg/Documents/Landmark Trajectories/target_points.txt'
        with open(file_path, 'w') as f:
            for coord in coordinates:
                f.write(f'{coord[0]}, {coord[1]}, {coord[2]}\n')
        
        ui.messageBox(f'Coordinates exported successfully to {file_path}.')

    except Exception as e:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

# Run the script
run(None)
