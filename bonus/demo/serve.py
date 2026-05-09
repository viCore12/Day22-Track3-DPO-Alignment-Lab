"""
Gradio Chatbot Interface cho Customer-service Chatbot.
Lưu ý: Để chạy cần cài đặt: pip install gradio unsloth
"""
import gradio as gr
import time

# Giả lập model response để demo khi chưa chạy train.py (POC stage)
def chat_with_bot(message, history):
    # Dummy logic to show the vibe of the bot
    lower_msg = message.lower()
    
    if "trà sữa" in lower_msg:
        response = "Dạ quán em chuyên phục vụ Specialty Coffee và các loại trà trái cây thanh mát, hiện quán không có trà sữa ạ. Mình có muốn thử món Trà Vải Hoa Hồng đang best-seller bên em không?"
    elif "mấy giờ" in lower_msg or "đóng cửa" in lower_msg:
        response = "Dạ quán em mở cửa từ 7:00 sáng đến 10:30 tối mỗi ngày ạ. Mình dự định ghé quán lúc mấy giờ để em sắp xếp chỗ ngồi cho mình ạ?"
    elif "wifi" in lower_msg:
        response = "Dạ pass wifi của Hanoi Coffee Roastery là 'caphengondohai' viết liền không dấu ạ. Mình kết nối thử xem có được không nhé!"
    else:
        response = "Dạ cảm ơn mình đã nhắn tin cho Hanoi Coffee Roastery. Quán em chuyên các dòng cà phê ủ lạnh và pour over. Mình đang muốn tìm thức uống vị thế nào để em tư vấn ạ?"
    
    # Simulate streaming
    for i in range(len(response)):
        time.sleep(0.01)
        yield response[:i+1]

# Cấu hình giao diện Gradio
theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone"
)

demo = gr.ChatInterface(
    chat_with_bot,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Nhắn tin cho quán ở đây...", container=False, scale=7),
    title="☕ Hanoi Coffee Roastery Support",
    description="Demo Chatbot CSKH sau khi fine-tune DPO. Thử hỏi về: giờ mở cửa, pass wifi, hoặc gọi một ly trà sữa.",
    theme=theme,
    examples=["Quán mấy giờ đóng cửa vậy?", "Cho mình pass wifi", "Quán có bán trà sữa trân châu đường đen không?"]
)

if __name__ == "__main__":
    print("Khởi động server Demo Chatbot...")
    demo.launch(share=False)
