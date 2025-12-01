# 中文
sglang的kernel模块死活编译不出来  
失败原因: killed，大概是在编译cuda相关的模块的时候，它好像强行编译sm90和sm100架构，无论你本地是什么arch  
所以干脆不编译整个模块了，把自己想要的模块抠出来，搞个python接口单独测试，美滋滋  
# English
I couldn't compile the kernel module in sglang no matter what I tried.
The failure reason: killed. It seems that when compiling the CUDA-related modules,
it's forcibly compiling for sm90 and sm100 architectures regardless of what you have locally.

So I simply gave up compiling the whole module. Instead, I extracted the parts I wanted,
created a Python interface, and tested them separately. Works like a charm! 😎
