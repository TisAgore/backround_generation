import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from typing import Tuple, List, Dict
import os
import random
import json
import csv

from deform_aug import new_local as defomation_augmintaion 


class BackgroundMethods:
    """Класс для создания различныхх методов для фоновых шаблонов"""
    
    @staticmethod
    def solid_color(width: int, height: int, color: Tuple[int, int, int]) -> np.ndarray:
        """Сплошной фон"""
        return np.full((height, width, 3), color, dtype=np.uint8)
    
    @staticmethod
    def gradient(width: int, height: int, 
                colors: List[Tuple[int, int, int]], 
                direction: str = 'vertical') -> np.ndarray:
        """Градиентный фон"""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height if direction == 'vertical' else width):
            ratio = i / (height - 1) if direction == 'vertical' else i / (width - 1)
            
            if len(colors) == 1:
                color = colors[0]
            else:
                segment = ratio * (len(colors) - 1)
                idx = int(segment)
                idx = min(idx, len(colors) - 2)
                local_ratio = segment - idx
                color = tuple(
                    int(colors[idx][j] + (colors[idx + 1][j] - colors[idx][j]) * local_ratio)
                    for j in range(3)
                )
            
            if direction == 'vertical':
                bg[i, :] = color
            else:
                bg[:, i] = color
        
        return bg
    
    @staticmethod
    def stripes(width: int, height: int, 
                colors: List[Tuple[int, int, int]], 
                direction: str = 'vertical',
                stripe_width: int = None) -> np.ndarray:
        """Фон в полосках"""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        
        if stripe_width is None:
            stripe_width = random.randint(30, 80)
        
        num_stripes = (width if direction == 'vertical' else height) // stripe_width + 2
        
        for i in range(num_stripes):
            color = colors[i % len(colors)]
            start = i * stripe_width
            end = start + stripe_width
            
            if direction == 'vertical':
                bg[:, start:min(end, width)] = color
            else:
                bg[start:min(end, height), :] = color
        
        return bg
    
    @staticmethod
    def vintage_paper(width: int, height: int) -> np.ndarray:
        """Фон с текстурой 'винтажной бумаги' (по факту шум добавленный на слошной фон)"""
        bg = np.full((height, width, 3), (235, 225, 200), dtype=np.uint8)
        
        noise = np.random.normal(0, 15, (height, width, 3))
        bg = np.clip(bg.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return bg

    @staticmethod
    def grid(width: int, height: int,
            thickness: int,
            cell_size: int,
            color: Tuple[int, int, int] = (0, 0, 0),
            bg_color: Tuple[int, int, int] = (200,225,225),
            ) -> np.ndarray:
        """Фон в сеточку (в клеточку)"""

        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:, :] = bg_color
                    
        if thickness is None:
            thickness = random.randint(1, 5)

        if cell_size is None:
            cell_size = random.randint(height//5, height//2)

        rows = height // cell_size
        columns = width // cell_size  
                    
        for i in range(rows):
            start = (i + 1) * cell_size
            end = start + thickness
            
            bg[start:min(end, height), :] = color

        for i in range(columns):
            start = (i + 1) * cell_size
            end = start + thickness
                    
            bg[:, start:min(end, width)] = color

        if random.choice([True, False]):
            start = random.choice([-1, 1]) * random.randint(1, 5) * cell_size + 3
            end = start + thickness + 1
            bg[:, start:min(end, width)] = (200, 0, 0)
        return bg

    @staticmethod
    def lines(width: int, height: int,
            thickness: int,
            line_gap: int,
            color: Tuple[int, int, int],
            bg_color: Tuple[int, int, int]
            ) -> np.ndarray:
        """Фон в линейку (здесь линии отделяют сплошной фон а не чередуются цвета как в stripes)"""

        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:, :] = bg_color

        if thickness is None:
            thickness = random.randint(1, 5)

        if line_gap is None:
            line_gap = random.randint(height // 5, height // 2)

        rows = height // line_gap

        for i in range(rows):
            start = (i + 1) * line_gap
            end = start + thickness

            bg[start:min(end, height), :] = color

        if random.choice([True, False]):
            start = random.choice([-1, 1]) * random.randint(1, 5) * line_gap + 3
            end = start + thickness + 1
            bg[:, start:min(end, width)] = (200, 0, 0)
        return bg


class BackgroundTemplates:
    """Класс для добавления шаблонов используя методы из класс BaclgroundMethods"""

    # Шаблоны
    TEMPLATES = {
        'solid_light': {
            'name': 'Light Solid',
            'type': 'solid_color',
            'params': {'color': (240, 240, 250)}
        },
        'stripes_warm': {
            'name': 'Warm Stripes',
            'type': 'stripes',
            'params': {
                'colors': [(255,225,195), (255,215,185)],
                'direction': 'horizontal',
                'stripe_width': 100
            }
        },
        'vintage': {
            'name': 'Vintage Paper',
            'type': 'vintage_paper',
            'params': {}
        },
        'blue_grid': {
            'name': 'Blue Grid',
            'type': 'grid',
            'params': {
                'color': (15,215,240),
                'bg_color': (200,225,225),
                'thickness': 3,
                'cell_size': None
                }
        },
        'black_grid': {
            'name': 'Black Grid',
            'type': 'grid',
            'params': {
                'color': (100,100,110),
                'bg_color': (250,250,255),
                'thickness': 3,
                'cell_size': None
                }
        },
        'red_grid': {
            'name': 'Red Grid',
            'type': 'grid',
            'params': {
                'color': (100,100,110),
                'bg_color': (250,225,225),
                'thickness': 3,
                'cell_size': None
                }
        },
        'blue_lines': {
            'name': 'Lines',
            'type': 'lines',
            'params': {
                'color': (15,215,240),
                'bg_color': (200,225,225),
                'thickness': 3,
                'line_gap': None
                }
        },
        'black_lines': {
            'name': 'Black Lines',
            'type': 'lines',
            'params': {
                'color': (100,100,110),
                'bg_color': (250,250,255),
                'thickness': 3,
                'line_gap': None
                }
        },
        'red_lines': {
            'name': 'Red Lines',
            'type': 'lines',
            'params': {
                'color': (100,100,110),
                'bg_color': (250,225,225),
                'thickness': 3,
                'line_gap': None
                }
        }
    }
    
    @classmethod
    def get_template_names(cls) -> List[str]:
        """Получить лист (массив) с названиями всех шаблонов"""
        return list(cls.TEMPLATES.keys())
    
    @classmethod
    def get_random_template(cls) -> Dict:
        """Получить случайную конфигурацию (сами атрибуты) шаблоны"""
        template_name = random.choice(list(cls.TEMPLATES.keys()))
        return cls.get_template(template_name)
    
    @classmethod
    def get_template(cls, template_name: str) -> Dict:
        """Получить конфигурацию (сами атрибуты) шаблона"""
        return cls.TEMPLATES.get(template_name)
    
    @classmethod
    def create_background(cls, template_name: str, 
                          width: int, height: int) -> np.ndarray:
        """Создать фон из шаблона"""
        template = cls.get_template(template_name)
        if template is None:
            template = cls.get_random_template()
        
        bg_type = template['type']
        params = template['params'].copy()
        
        # Add width and height if not present
        if 'width' not in params:
            params['width'] = width
        if 'height' not in params:
            params['height'] = height
        
        # Call the appropriate static method
        method = getattr(BackgroundMethods, bg_type)
        return method(**params)


class TextImageGenerator:
    """Класс для генерации изображений с текстом и различными фонами"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.image = None
        self.pil_image = None
        self.draw = None
        self.current_background_template = None
        
    def _update_pil_image(self):
        if self.pil_image is None:
            if self.image is None:
                self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.pil_image = Image.fromarray(self.image)
            self.draw = ImageDraw.Draw(self.pil_image)
    
    def _update_numpy_image(self):
        if self.pil_image is not None:
            self.image = np.array(self.pil_image)
            self.draw = ImageDraw.Draw(self.pil_image)
    
    def apply_background_template(self, template_name: str) -> 'TextImageGenerator':
        """Применить фоновый шаблон"""
        self.image = BackgroundTemplates.create_background(
            template_name, self.width, self.height
        )
        self.pil_image = Image.fromarray(self.image)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.current_background_template = template_name
        return self

    def apply_base_background(self) -> 'TextImageGenerator':
        """Применить базовый фоновый шаблон (использую solid light, т.к. аугментация Юры работает только со сплошным фоном)"""
        templates = BackgroundTemplates.get_template_names()
        base_template = ''
        for i in templates:
            if i == 'solid_light':
                base_template = i
        return self.apply_background_template(base_template)
    
    def apply_random_background(self) -> 'TextImageGenerator':
        """Применить случайный фоновый шаблон"""
        template_name = random.choice(BackgroundTemplates.get_template_names())
        return self.apply_background_template(template_name)
    
    def add_text(self, 
                text: str,
                position: Tuple[int, int],
                font_path: str,
                font_size: int = 100,
                color: Tuple[int, int, int] = (255, 255, 255),
                stroke_width: int = 0,
                stroke_color: Tuple[int, int, int] = (0, 0, 0),
                anchor: str = 'lt',
                spacing: int = 0,
                align: str = 'left',
                shadow: bool = False,
                shadow_color: Tuple[int, int, int] = (168,168,168),
                shadow_offset: Tuple[int, int] = (2, 2)) -> 'TextImageGenerator':
        """
        Добавляет текст к фону (изображению)
        """
        self._update_pil_image()
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Warning: Could not load font: {e}")
            return self
        
        color_tuple = tuple(color)
        stroke_color_tuple = tuple(stroke_color) if stroke_width > 0 else None
        
        # тени
        if shadow:
            shadow_pos = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
            self.draw.text(
                shadow_pos, text, font=font, fill=tuple(shadow_color),
                anchor=anchor, spacing=spacing, align=align
            )
        
        # обводка текста
        if stroke_width > 0 and stroke_color_tuple:
            self.draw.text(
                position, text, font=font, fill=stroke_color_tuple,
                stroke_width=stroke_width, stroke_fill=stroke_color_tuple,
                anchor=anchor, spacing=spacing, align=align
            )
            self.draw.text(
                position, text, font=font, fill=color_tuple,
                anchor=anchor, spacing=spacing, align=align
            )
        else:
            self.draw.text(
                position, text, font=font, fill=color_tuple,
                anchor=anchor, spacing=spacing, align=align
            )
        
        self._update_numpy_image()
        return self
    
    def save(self, filepath: str):
        if self.pil_image is not None:
            self.pil_image.save(filepath)
            print(f"Image saved to {filepath}")
    
    def show(self, figsize: Tuple[int, int] = (10, 8)):
        if self.image is not None:
            plt.figure(figsize=figsize)
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
    
    def get_image(self) -> np.ndarray:
        return self.image


def generate_text_variations(text_line: str,
                            output_dir: str,
                            text_num: int,
                            num_variations: int = 5,
                            width: int = 1200,
                            height: int = 800,
                            show_preview: bool = False,
                            is_deform_aug: bool = True):
    """
    Генерация указанного кол-ва различных вариантов фонов и текста
    
    Args:
        text_lines: Text string
        output_dir: Directory to save images
        font_path: Path to font file
        num_variations: Number of variations to generate
        width: Image width
        height: Image height
        show_preview: Show preview of each image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    font_dir = "fonts"
    font_size = 100
    
    # Color palettes for text
    text_colors = [
        (255, 30, 30),  # Red
        (20,120,200),  # Light Blue
        (0, 40, 140),  # Blue
        (0, 0, 100),      # Dark Blue
        (0, 0, 0)        # Black
    ]
    
    # Generate variations
    for i in range(num_variations):
        font_name = f'fonts/{random.choice(os.listdir(font_dir))}'

        generator = TextImageGenerator(width, height)
        generator.apply_base_background()

        text_pos = random.randint(1, 3)
        y_pos = height // 2
        if text_pos == 1:
            x_pos = font_size * 2
            anchor = random.choice(['lm', 'lt', 'lb'])
        elif text_pos == 2:
            x_pos = width // 2
            anchor = random.choice(['mm', 'mt', 'mb'])
        else:
            x_pos = width - font_size * 2
            anchor = random.choice(['rm', 'rt', 'rb'])

        if 't' in anchor:
            y_pos = height // 2 - font_size
        if 'b' in anchor:
            y_pos = height // 2 + font_size
        
        generator.add_text(
            text=text_line,
            position=(x_pos, y_pos),
            font_path=font_name,
            font_size=font_size,
            color=(0, 0, 0),
            stroke_width=0,
            stroke_color=(0, 0, 0),
            anchor=anchor,
            shadow=False
        )

        text_color = random.choice(text_colors)

        if is_deform_aug:
            base_augment = defomation_augmintaion(generator.get_image(), times=1)
            base_augment = base_augment[0]
            base_augment = np.array(base_augment)

            aug_text = np.all(base_augment == 0, axis=2)
            color_text_img = base_augment.copy()
            color_text_img[aug_text] = text_color
            mask_for_bg = np.all(color_text_img == text_color, axis=2)

            # Save image with aug
            generator.apply_random_background()
            result_img = generator.get_image()
            result_img[mask_for_bg] = text_color
            generator.image = result_img
            generator.pil_image = Image.fromarray(result_img)
            output_path = os.path.join(output_dir, f'{text_num}_aug_{i}.png')
            generator.save(output_path)

        base_text = np.all(generator.get_image() == 0, axis=2)
        color_text_base_img = generator.get_image().copy()
        color_text_base_img[base_text] = text_color
        mask_for_bg2 = np.all(color_text_base_img == text_color, axis=2)

        # Save image without aug
        generator.apply_random_background()
        result_img = generator.get_image()
        result_img[mask_for_bg2] = text_color
        generator.image = result_img
        generator.pil_image = Image.fromarray(result_img)
        output_path = os.path.join(output_dir, f'{text_num}_{i}.png')
        generator.save(output_path)
        
        # Показать превью вариации
        if show_preview:
            print(f"\n--- Variation {i+1} ---")
            print(f"Background: {generator.current_background_template}")
            generator.show(figsize=(8, 5))
    return


if __name__ == "__main__":
    text_file = list()
    with open('test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            text_file.append(line)
            # if len(text_file) > 10:
            #     break
    output_dir = 'synthetic'
    i = 0
    is_deform_aug = True
    num_variations = 2
    annotations_json = []
    csv_file = open('dataset.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['path', 'text'])
    for text_line in text_file:
        text_line = text_line.split('\n')[0]
        i += 1
        generate_text_variations(
                    text_line=text_line,
                    output_dir=output_dir,
                    num_variations=num_variations,
                    width=int(len(text_line) * 0.75) * 100,
                    height=250,
                    show_preview=False,
                    text_num=i,
                    is_deform_aug=is_deform_aug
                )

        for variant in range(num_variations):
            filename = f'{output_dir}/{i}_{variant}.png'
            annotations_json.append({"file_name": filename, "ocr": text_line})
            csv_writer.writerow([filename, text_line])
            if is_deform_aug:
                filename = f'{output_dir}/{i}_aug_{variant}.png'
                annotations_json.append({"file_name": filename, "ocr": text_line})
                csv_writer.writerow([filename, text_line])
        # csv

        with open('annotations.json', 'w', encoding='utf-8') as f:
            json.dump(annotations_json, f, ensure_ascii=False, indent=2)
        # json

    csv_file.close()
    
