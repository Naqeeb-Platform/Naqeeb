from PIL import ImageFont, ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display

def process_image(image_path, labels_dir, arabic_font, arabic_font_size):
    # Load the image
    img = cv2.imread(image_path)

    txt_file = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    if not os.path.exists(txt_file):
        return img  # Return the original image if no text file found

    with open(txt_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            box_data = line.strip().split(" ")
            label = int(box_data[0])  # Extract label
            x, y, w, h = map(float, box_data[1:])  # Extract box coordinates
            x1, y1 = int((x - w / 2) * img.shape[1]), int((y - h / 2) * img.shape[0])
            x2, y2 = int((x + w / 2) * img.shape[1]), int((y + h / 2) * img.shape[0])

            label_text = f"Label: {labels_new[str(label)]}"  # Construct label text
            reshaped_text = arabic_reshaper.reshape(label_text)
            bidi_text = get_display(reshaped_text)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 255), 2)  # Changed color to a cool blue (RGB: 0, 191, 255)

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            text_bbox = draw.textbbox((x1, y1 - arabic_font_size - 10), bidi_text, font=arabic_font)
            draw.rectangle(text_bbox, fill=(255, 128, 0))  # Fill under the text with the same blue color
            draw.text((x1, y1 - arabic_font_size - 10), bidi_text, font=arabic_font, fill=(255, 255, 255))

            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img
