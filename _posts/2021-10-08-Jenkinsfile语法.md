---
title: Jenkinsfile语法
date: 2021-10-08 10:10:00 UTC
categories:
- 持续集成/jenkins
tags: jenkins
---
# Jenkinsfile 语法
## 框架
```
pipeline {
    # 声明流水线标识
    agent{
        # 必须 流水线执行位置
    }
    environment{
        # 可选 设置环境变量
    }
    triggers{
        # 可选 流水线触发器，支持三种触发器cron、pollSCM、upstream
    }
    libraries{
        # 可选 
    }
    options{
        # 可选 用来配置Jenkins应用自认的一些配置项
    }
    parameters{
        # 可选 提供用户在触发pipeline时应提供的参数列表
    }
    tools{
        # 可选 定义部署流程中常用的一些工具
    }
    stages{
        # 必须 有且仅有一个 此关键字用于表示流水线各个步骤的声明
        stage('1阶段名称'){
            # 必须 至少存在一个 表示实际构建的阶段
            agent{
                # 可选 流水线该节点的执行位置
            }
            environment{
                # 可选 该节点的环境变量
            }
            tools{
                # 可选 该节点需要使用的工具
            }
            input{
                # 可选 使用该指令来提示输入
            }
            when{
                # 可选 根据流水线给定的条件决定是否应该执行
            }
            steps{
                # 必须 有且仅有一个 标识阶段中具体的构建步骤
            }
            post{
                # 可选 定义该阶段执行完成之后的结果
            }
        }
        stage('2阶段名称'){

        }
        post{
            # 可选 定义整条流水线执行之后的结果
        }
    }
} 
```
## 基础模板
vscode中安装JenkinsFile Support插件后，输入pipe后将出现一下模板。

```
pipeline{
    agent{
        label "node"
    }
    stages{
        stage("A"){
            steps{
                echo "========executing A========"
            }
            post{
                always{
                    echo "========always========"
                }
                success{
                    echo "========A executed successfully========"
                }
                failure{
                    echo "========A execution failed========"
                }
            }
        }
    }
    post{
        always{
            echo "========always========"
        }
        success{
            echo "========pipeline executed successfully ========"
        }
        failure{
            echo "========pipeline execution failed========"
        }
    }
}

```
### pipeline
所有的语言内容都需要包含在该语法块中，没有特殊意义。
```
pipeline{
    ...
}
```
### agent
<font color='pink'>必须</font>
此关键词用于表示当前流水线将要执行的位置，也可以表明docker容器的构建。

举例： 构建一个node编译环境，并将环境放入docker中。
```
pipeline {
    agent {
        docker {
        	lable 'docker'
            image 'registry.cn-hangzhou.aliyuncs.com/eryajf/node:11.15'
        }
    }
    stages {
        stage('Build') { 
            steps {
                sh 'npm install --registry=https://registry.npm.taobao.org' 
            }
        }
    }
}
```


