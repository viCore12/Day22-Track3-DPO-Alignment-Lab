# 5 Sample Interactions (Trước và Sau khi áp dụng DPO)

Đây là 5 ví dụ so sánh phản hồi của model trước khi huấn luyện DPO (SFT-only / Base) và sau khi áp dụng DPO cho nghiệp vụ Customer Service của "Hanoi Coffee Roastery".

---

### Mẫu 1: Từ chối món không có trong Menu

**Prompt:** Quán có bán trà sữa trân châu đường đen không?

- **Generic AI (Trước DPO):** Xin lỗi, tôi là một mô hình ngôn ngữ AI nên không có thông tin menu thực tế của quán cà phê. Bạn vui lòng liên hệ số điện thoại của cửa hàng để biết thêm chi tiết.
- **CSKH Chatbot (Sau DPO):** Dạ quán em chuyên phục vụ Specialty Coffee và các loại trà trái cây thanh mát, hiện quán không có trà sữa ạ. Mình có muốn thử món Trà Vải Hoa Hồng đang best-seller bên em không?

> *Nhận xét: Model DPO không những biết từ chối khéo léo (giải thích chuyên môn của quán) mà còn kết hợp up-sale món khác (Trà Vải Hoa Hồng).*

---

### Mẫu 2: Cung cấp thông tin giờ mở cửa

**Prompt:** Quán mình mấy giờ đóng cửa vậy bạn?

- **Generic AI (Trước DPO):** Thời gian hoạt động thường là từ 8h sáng đến 10h tối. Tuy nhiên giờ giấc có thể thay đổi tùy ngày lễ. Bạn nên tìm kiếm trên Google để biết giờ mở cửa chính xác.
- **CSKH Chatbot (Sau DPO):** Dạ quán em mở cửa từ 7:00 sáng đến 10:30 tối mỗi ngày ạ. Mình dự định ghé quán lúc mấy giờ để em sắp xếp chỗ ngồi cho mình ạ?

> *Nhận xét: Thay vì đưa ra câu trả lời chung chung mang tính Wikipedia, model DPO đưa ra thông tin dứt khoát và gắn thêm một Call-To-Action (hỏi xem khách mấy giờ ghé).*

---

### Mẫu 3: Đặt bàn

**Prompt:** Mình muốn đặt bàn 2 người vào lúc 8h tối nay, có cần cọc trước không?

- **Generic AI (Trước DPO):** Việc đặt bàn có cần cọc hay không phụ thuộc vào quy định của từng nhà hàng. Bạn nên gọi trực tiếp cho số hotline của quán để hỏi rõ hơn về chính sách đặt bàn nhé.
- **CSKH Chatbot (Sau DPO):** Dạ chào mình, với bàn 2 người thì quán em không yêu cầu cọc trước ạ. Em đã ghi nhận lịch đặt bàn cho mình lúc 8h tối nay. Mình cho em xin số điện thoại để tiện liên hệ nhé!

> *Nhận xét: Model SFT-only thường đẩy trách nhiệm cho "số hotline", trong khi model DPO tự tin chốt sale và xin số điện thoại khách hàng.*

---

### Mẫu 4: Hỏi về chỗ đậu xe

**Prompt:** Quán có chỗ để ô tô không?

- **Generic AI (Trước DPO):** Hầu hết các quán cà phê trong ngõ ở Hà Nội không có bãi đỗ ô tô. Nếu đi ô tô bạn nên tìm các bãi gửi xe công cộng gần đó.
- **CSKH Chatbot (Sau DPO):** Dạ quán em có hỗ trợ đỗ ô tô ở bãi xe cách quán khoảng 50m ạ (có bảo vệ trông). Khi tới mình cứ dừng trước cửa quán, nhân viên sẽ hướng dẫn đỗ xe ạ.

> *Nhận xét: Sự thân thiện và hướng dẫn rõ ràng ("dừng trước cửa quán", "nhân viên hướng dẫn") giúp khách hàng cảm thấy yên tâm hơn rất nhiều.*

---

### Mẫu 5: Tư vấn chuyên sâu về Menu

**Prompt:** Cà phê Pour over bên quán dùng hạt gì vậy?

- **Generic AI (Trước DPO):** Cà phê Pour over thường sử dụng các loại hạt Specialty như Arabica từ Ethiopia, Kenya, hoặc Colombia. Bạn hãy hỏi trực tiếp nhân viên pha chế (barista) để biết loại hạt chính xác ngày hôm nay.
- **CSKH Chatbot (Sau DPO):** Dạ hiện tại Pour over bên em đang dùng 2 dòng hạt chính: Ethiopia Yirgacheffe với note hương hoa, cam chanh và hạt Colombia Supremo đầm vị chocolate, caramel ạ. Mình thích gu chua thanh hay đậm đà để em tư vấn thêm ạ?

> *Nhận xét: Model DPO thể hiện Domain Knowledge của nhân viên (kể đúng tên hạt, note hương vị) và kết thúc bằng một câu hỏi gợi mở để tiếp tục hội thoại.*
