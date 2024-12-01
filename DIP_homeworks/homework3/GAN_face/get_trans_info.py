import cv2
import numpy as np
import gradio as gr
import face_alignment as fa

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None
module_run = False  # 全局变量，用于跟踪 export_points_to_file 是否运行过

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行点的变换
def get_mapped_point(image, source_pts, target_pts):
    fp = fa.FaceAlignment(fa.LandmarksType.TWO_D, flip_input=False)
    preds = fp.get_landmarks(image)
    li = list(preds[0])
    list_in_int = [[int(num) for num in pair] for pair in li]
    for i in range(len(source_pts)):
        delt = source_pts[i] - np.array(list_in_int)
        nearset = list_in_int[np.argmin(np.sum((delt * delt), axis=1))]
        d_x = nearset[0] - source_pts[i][0]
        d_y = nearset[1] - source_pts[i][1]
        source_pts[i] = nearset
        target_pts[i] = [target_pts[i][0] + d_x, target_pts[i][1] + d_y]
    print(source_pts)
    print(target_pts)
    return source_pts, target_pts

def run_new_drag():
    global points_src, points_dst, image
    points_src, points_dst = get_mapped_point(image, points_src, points_dst)
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    return marked_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 导出点信息到文件
def export_points_to_file():
    global points_src, points_dst, module_run
    with open("info_p.py", "w") as f:
        f.write("points_src = " + str(points_src) + "\n")
        f.write("points_dst = " + str(points_dst) + "\n")
        f.write("module_run = True\n")
    return "Points exported to info_p.py"

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")
    output_button = gr.Button("Output Points to File")
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_new_drag, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    # 点击输出按钮，导出点信息到文件
    output_button.click(export_points_to_file, None, None)
    # 添加一个回调函数来检查 should_close 标志


def launch_app():
    demo.launch()

if __name__ == "__main__":
    launch_app()