import models_io.saved_model_pars.baseline_weights as BW
import levels.da_Statistical_Iris.hot_weights_4 as HW
import levels.da_Pretrained_Iris.Refined_2 as R2W
import levels.da_Pretrained_Iris.Refined_3 as R3W

# --------------------------------

from pathlib import Path

# 1. Get the path of the current script file
script_path = Path(__file__).resolve()
# 2. Get the directory containing the script
script_dir = script_path.parent

FIG_VERSION = 3
FIG_MINOR = 0

if FIG_VERSION == 0:
    VER_NAME = "Baseline"
    WEIGHTS = [BW.W1, BW.W2, BW.W3]
    BIASES = [BW.b1, BW.b2, BW.b3]
elif FIG_VERSION == 1:
    VER_NAME = "Refined_1"
    WEIGHTS = [HW.W1, HW.W2, HW.W3]
    BIASES = [HW.b1, HW.b2, HW.b3]
elif FIG_VERSION == 2:
    VER_NAME = "Refined_2"
    WEIGHTS = [R2W.W1, R2W.W2, R2W.W3]
    BIASES = [R2W.b1, R2W.b2, R2W.b3]
else: # FIG_VERSION == 3:
    VER_NAME = "Refined_3"
    WEIGHTS = [R3W.W1]
    BIASES = [R3W.b1]


from visual_iris_UI import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    canvas_widget = VisualIrisApp(WEIGHTS, BIASES, FIG_MINOR)
    
    setup_interactive_save(canvas_widget)
    canvas_widget.show()

    CLASSES = ['Setosa', 'Versicolor', 'Virginica']

    # the relative path to the subfolder and file
    relative_subfolder_path = Path(f"../../images/{CLASSES[FIG_MINOR]}_{VER_NAME}.png")
    # Combine them to get the absolute path to the data file
    file_path = str(script_dir / relative_subfolder_path)

    print(f"Saving Canvas to: {file_path}")
    do_print_canvas(file_path, canvas_widget)

    sys.exit(app.exec())