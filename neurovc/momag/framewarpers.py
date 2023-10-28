__author__ = "Cosmas Heiss, Philipp Flotho"

import moderngl
import numpy as np
from scipy.interpolate import griddata
import cv2


def warp_image_pc(img, flow):
    m, n = img.shape[:2]
    xi, yi = np.meshgrid(np.arange(n).astype(np.float), np.arange(m).astype(np.float))

    tmp = np.empty(img.shape, img.dtype)
    for i in range(3):
        griddata_result = griddata((xi.flatten() + flow[:, :, 0].flatten(),
                                    yi.flatten() + flow[:, :, 1].flatten()),
                                   img[:, :, i].flatten(), (xi.flatten(), yi.flatten()),
                                   method="linear", fill_value=0)
        tmp[:, :, i] = np.reshape(griddata_result, (m, n))

    return tmp


def warp_image_pc_single(img, flow):
    m, n = img.shape[:2]
    xi, yi = np.meshgrid(np.arange(n).astype(np.float), np.arange(m).astype(np.float))

    tmp = np.empty(img.shape, img.dtype)

    griddata_result = griddata((xi.flatten() + flow[:, :, 0].flatten(),
                                yi.flatten() + flow[:, :, 1].flatten()),
                               img[:, :].flatten(), (xi.flatten(), yi.flatten()),
                               method="nearest", fill_value=0)
    tmp[:, :] = np.reshape(griddata_result, (m, n))

    return tmp


def warp_image_backwards(img, flow):
    h, w = flow.shape[:2]
    # flow = -flow
    tmp_flow = np.empty(flow.shape, np.array(flow).dtype)
    np.copyto(tmp_flow, flow)
    tmp_flow[:, :, 0] += np.arange(w)
    tmp_flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, tmp_flow, None, cv2.INTER_LINEAR)
    return res


# moderngl-based fast forward warping, author: Cosmas Heiß
class OnlineFrameWarper:
    def __init__(self, image_size):
        self.image_size = image_size
        self.strip_indices = self.generate_triangle_strip_index_array()
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_vert;
                in vec3 in_color;
                in float in_depth;
                out vec3 v_color;


                void main() {
                    v_color = in_color;
                    gl_Position = vec4(in_vert, in_depth, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;

                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            """,
        )
        dummy_vertices = self.get_dummy_vertices()
        self.vertex_buffer = self.ctx.buffer(dummy_vertices)
        self.vertex_array = self.ctx.vertex_array(self.prog, self.vertex_buffer, 'in_vert', 'in_color', 'in_depth')
        self.frame_buffer = self.ctx.simple_framebuffer(image_size[::-1])

    def get_dummy_vertices(self):
        image = np.zeros((*self.image_size, 3))
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.image_size[1], endpoint=True), np.linspace(-1, 1, self.image_size[0], endpoint=True))
        displacements = np.stack((xx, yy), axis=2)
        depth = np.zeros(self.image_size)
        return self.get_into_vertex_buffer_shape(image, displacements, depth)

    def get_into_vertex_buffer_shape(self, image, displacements, depth):
        vertices = np.concatenate((displacements, image, depth[:, :, None]), axis=2)[self.strip_indices[:, 0], self.strip_indices[:, 1]]
        return vertices.astype('f4').tobytes()

    def generate_triangle_strip_index_array(self):
        out_indices_x = []
        out_indices_y = []
        for i in range(self.image_size[1]-1):
            is_reversed = int((i % 2) * (-2) + 1)
            out_indices_x.append(np.arange(self.image_size[0]).repeat(2)[::is_reversed])
            out_indices_y.append(np.tile(np.array([i, i+1]), self.image_size[0]))

        out_indices_x = np.concatenate(out_indices_x).astype(int)
        out_indices_y = np.concatenate(out_indices_y).astype(int)
        return np.stack((out_indices_x, out_indices_y), axis=1)

    def read_frame_buffer(self):
        return np.frombuffer(self.frame_buffer.read(), 'uint8').reshape(self.image_size[0], self.image_size[1], -1)

    def pixel_to_screenspace_coords(self, displacements):
        out = np.zeros_like(displacements)
        out[:, :, 1] = displacements[:, :, 0] * (2.0 / (self.image_size[0] - 1)) - 1.0
        out[:, :, 0] = displacements[:, :, 1] * (2.0 / (self.image_size[1] - 1)) - 1.0
        return out

    def warp_image(self, image, displacements, depth):
        assert image.dtype == displacements.dtype == depth.dtype == float
        assert np.all(np.logical_and(image <= 1.0, image >= 0.0))
        assert image.shape[:2] == displacements.shape[:2] == depth.shape == self.image_size

        displacements = self.pixel_to_screenspace_coords(displacements)
        if depth.max() != depth.min():
            depth = - 0.99 * (depth - depth.min()) / (depth.max() - depth.min())

        self.frame_buffer.use()
        self.frame_buffer.clear(0.0, 0.0, 0.0, 0.0)

        self.vertex_buffer.write(self.get_into_vertex_buffer_shape(image, displacements, depth))

        self.vertex_array.render(moderngl.TRIANGLE_STRIP)

        return self.read_frame_buffer()[:, :, :3]

    def warp_image_uv(self, image, uv):
        m, n = uv.shape[0:2]

        x, y = np.meshgrid(np.arange(n).astype(float), np.arange(m).astype(float))

        tmp_flow = np.empty(uv.shape, float)
        tmp_flow[:, :, 0] = y + uv[:, :, 1]
        tmp_flow[:, :, 1] = x + uv[:, :, 0]

        return (self.warp_image(np.array(image).astype(float) / 255.0, tmp_flow, np.zeros((m, n), float)))
