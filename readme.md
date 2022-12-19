
<!-- ABOUT THE PROJECT -->
## Style Tranfer Project

Style Transfer là một bài toán biến đổi hình hành theo các style mong muốn. Trong bài toán này, mình sử dụng model VGG19 để lấy thông tin features tại từng convolution layer.

###  Style Tranfer Technical

+ Lấy features: Sử dụng VGG19 để lấy thông tin features tại từng convolution layer.
+ Xây dựng hàm loss:  wieght_1 x content loss (bức ảnh gốc so với ảnh mục tiêu) + wieght_2 x style_loss (bức ảnh style so với ảnh mục tiêu)
+ Update gradient ảnh target


### Built  Environment

Setup môi trường và chạy server

<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

*python
  ```python 3.8.5
  ```

### Installation



1. Clone the repo
   ```
   git clone
   ```
3. tạo môi trường `conda` hoặc `virtualenv`, ví dụ tên môi trường: `venv`
   *conda
   ```
   conda create -n venv python=3.8.5
   ```
   *virtualenv
   ```
   virtualenv venv
   ```

3. Activate môi trường:
   *conda
   ```
   conda activate venv
   ```
   *virtualenv
   ```
   source venv\bin]activate

4. Tải thư viện:
    ```
    pip install -r requirements/dev.txt
    ```
5. chạy server:
    *step1:
    ```
    cd src
    ```
    *step2:
    ```
    sh start_script.sh
    ```


<!-- USAGE EXAMPLES -->
## Usage

Sau quá trình cài thư viên và chạy server hoàn tất, bạn có thể vào đường link: ```http://0.0.0.0:8000/docs``` để thử nghiệm.

<!-- CONTACT -->
## Contact

Manh Pham - pvm26042000@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Một số kiến thức liên quan để có thể thực hiện 
* [](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)