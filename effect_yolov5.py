import cv2
import torch
import numpy as np
import sys
import argparse

class FlameParticle:
    def __init__(self, x, y,color):
        self.x = x
        self.y = y
        self.size = np.random.randint(5, 12)
        self.alpha = 255
        self.lifespan = np.random.randint(100, 200)
        self.color = color
        self.is_alive = True
        self.vx = np.random.uniform(-0.3, 0.3)
        self.vy = np.random.uniform(-1, -0.5)
        self.gravity = 0.1
        self.turbulence = np.random.uniform(-0.1, 0.1)

    def update(self):
        if self.is_alive:
            #위치 및 크기 업데이트
            self.x += self.vx
            self.y += self.vy
            self.vy += self.gravity + self.turbulence
            self.size = max(self.size - 0.5, 1)

            #감소 효과
            self.alpha = int(self.alpha * (self.lifespan / 100))
            self.lifespan -= 2

            #사라짐
            if self.alpha <= 0 or self.size <= 1 or self.lifespan <= 0:
                self.is_alive = False
                
            #진동추가
            self.vx += self.turbulence
            self.vy += self.turbulence

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.size = np.random.randint(5, 12)
        self.alpha = 255
        self.lifespan = np.random.randint(50, 100)
        self.is_alive = True
        self.vx = np.random.uniform(-0.3, 0.3)
        self.vy = np.random.uniform(-1, -0.5)
        self.turbulence = np.random.uniform(-0.1, 0.1)

class FlameGenerator:
    def __init__(self, model_path, video_path, output_video_path, style, num_points=5, ratio=0.6):
        self.model = torch.hub.load('.', 'custom', path=model_path, force_reload=True, source='local')
        self.model.eval()
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.num_points = num_points
        self.style=style
        self.angle = 2 * np.pi / num_points
        self.ratio = ratio
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (self.frame_width, self.frame_height))
        self.flame_particles = []
        
    def generate_particles(self, frame, detections):
        for det in detections:
            bbox = det[:4]
            x1, y1, x2, y2 = map(int, bbox)
            for _ in range(20):
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                new_width = int((x2 - x1) * self.ratio)
                new_height = int((y2 - y1) * self.ratio)

                new_x = np.random.randint(center_x - new_width // 2, center_x + new_width // 2)
                new_y = np.random.randint(center_y - new_height // 2, center_y + new_height // 2)
                if self.style == 'star':
                    color = (
                        np.random.randint(50, 220),
                        np.random.randint(50, 220),
                        np.random.randint(50, 220)
                    )
                elif self.style == 'fire':
                    color = (
                        np.random.randint(0, 40),
                        np.random.randint(0, 150),
                        np.random.randint(210, 256),
                    )
                self.flame_particles.append(FlameParticle(new_x, new_y,color))
        
    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            flames = np.zeros(frame.shape, dtype=np.uint8)
            results = self.model(frame)
            detections = results.pred[0]
            self.generate_particles(frame, detections) 

            #star 이펙트
            if self.style == 'star':
                for particle in self.flame_particles:
                    if particle.is_alive:
                        points = []
                        for i in range(self.num_points * 2):
                            radius = particle.size if i % 2 == 0 else particle.size * 0.5
                            x = particle.x + radius * np.cos(i * self.angle)
                            y = particle.y + radius * np.sin(i * self.angle)
                            points.append((int(x), int(y)))
                        cv2.fillConvexPoly(flames, np.array(points), particle.color)
                        particle.update()
            
            #fire 이펙트
            elif self.style == 'fire':
                for particle in self.flame_particles:
                    if particle.is_alive:
                        cv2.circle(flames, (int(particle.x), int(particle.y)), int(particle.size), particle.color, -1)
                        particle.update()

            #영상 저장
            result = cv2.addWeighted(frame, 1, flames, 0.8, 0)
            self.out.write(result)

            #진행도
            current_frame += 1
            progress = current_frame / total_frames
            sys.stdout.write("\rProgress: {:.2%}".format(progress))
            sys.stdout.flush()

        self.cap.release()
        self.out.release()

#매개변수 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_opt()
    generator = FlameGenerator(model_path=args.model, video_path=args.video, output_video_path=args.output, style=args.style)
    generator.run()