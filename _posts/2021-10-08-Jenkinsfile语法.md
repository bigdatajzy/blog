---
title: Jenkinsfile语法
date: 2021-10-08 10:10:00 UTC
categories:
- 持续集成/jenkins
tags: jenkins
---
Jenkins流水线(Pipeline)是一套插件，支持在Jenkins中实现和集成持续交付流水线。本文详细介绍Jenkinsfile的语法规则和使用方法。

## Jenkinsfile基本结构

Jenkinsfile使用声明式语法定义整个CI/CD流水线。基本结构如下：

```groovy
pipeline {
    agent {
        // 定义流水线执行位置
    }
    
    stages {
        // 定义流水线各个阶段
        stage('构建') {
            steps {
                // 具体构建步骤
            }
        }
        
        stage('测试') {
            steps {
                // 测试步骤
            }
        }
        
        stage('部署') {
            steps {
                // 部署步骤
            }
        }
    }
    
    post {
        // 流水线完成后的操作
    }
}
```

## 核心组件详解

### agent（必需）

定义流水线或特定阶段的执行环境，可以是Jenkins节点、Docker容器等。

```groovy
// 在任意可用节点上执行
agent any

// 在特定标签的节点上执行
agent { label 'my-node' }

// 使用Docker容器
agent {
    docker {
        image 'node:14'
        label 'docker'
    }
}

// 不使用全局agent，仅在各stage中定义
agent none
```

### stages和stage（必需）

`stages`包含所有`stage`定义，每个`stage`代表流水线的一个阶段。

```groovy
stages {
    stage('编译') {
        steps {
            sh 'mvn compile'
        }
    }
    stage('测试') {
        steps {
            sh 'mvn test'
        }
    }
}
```

### steps（必需）

定义每个阶段具体执行的步骤。

```groovy
steps {
    echo '执行构建'
    sh 'npm install'
    sh 'npm run build'
}
```

### environment（可选）

设置环境变量，可在流水线全局或特定阶段内定义。

```groovy
pipeline {
    environment {
        GLOBAL_VAR = 'global-value'
    }
    
    stages {
        stage('示例') {
            environment {
                STAGE_VAR = 'stage-value'
            }
            steps {
                echo "全局变量: ${GLOBAL_VAR}"
                echo "阶段变量: ${STAGE_VAR}"
            }
        }
    }
}
```

### parameters（可选）

定义流水线参数，允许用户在触发时提供输入。

```groovy
parameters {
    string(name: 'BRANCH', defaultValue: 'main', description: '要构建的分支')
    choice(name: 'ENV', choices: ['dev', 'test', 'prod'], description: '部署环境')
    booleanParam(name: 'SKIP_TEST', defaultValue: false, description: '是否跳过测试')
}
```

### when（可选）

条件判断，决定是否执行某个阶段。

```groovy
stage('仅在主分支部署') {
    when {
        branch 'main'
    }
    steps {
        echo '部署到生产环境'
    }
}
```

### post（可选）

定义阶段或整个流水线完成后的操作。

```groovy
post {
    always {
        echo '无论成功失败都执行'
    }
    success {
        echo '成功后执行'
    }
    failure {
        echo '失败后执行'
        mail to: 'team@example.com', subject: '构建失败'
    }
}
```

## 实用模板

一个完整的Jenkinsfile示例：

```groovy
pipeline {
    agent {
        label "node"
    }
    
    environment {
        REPO_URL = 'https://github.com/example/repo.git'
    }
    
    parameters {
        string(name: 'BRANCH', defaultValue: 'main', description: '要构建的分支')
    }
    
    stages {
        stage('检出代码') {
            steps {
                git branch: params.BRANCH, url: REPO_URL
            }
        }
        
        stage('构建') {
            steps {
                sh 'npm install --registry=https://registry.npm.taobao.org'
                sh 'npm run build'
            }
        }
        
        stage('测试') {
            when {
                expression { return !params.SKIP_TEST }
            }
            steps {
                sh 'npm test'
            }
        }
        
        stage('部署') {
            steps {
                echo '部署应用'
            }
            post {
                success {
                    echo '部署成功'
                }
            }
        }
    }
    
    post {
        always {
            echo '清理工作区'
            cleanWs()
        }
        success {
            echo '流水线执行成功'
        }
        failure {
            echo '流水线执行失败'
        }
    }
}
```

## 高级功能

### parallel（并行执行）

```groovy
stage('并行任务') {
    parallel {
        stage('任务A') {
            steps {
                echo '执行任务A'
            }
        }
        stage('任务B') {
            steps {
                echo '执行任务B'
            }
        }
    }
}
```

### input（用户交互）

```groovy
stage('确认部署') {
    input {
        message "是否部署到生产环境?"
        ok "确认部署"
        parameters {
            string(name: 'VERSION', defaultValue: '1.0.0', description: '部署版本')
        }
    }
    steps {
        echo "部署版本: ${VERSION}"
    }
}
```

### 使用共享库

```groovy
@Library('my-shared-library') _

pipeline {
    agent any
    stages {
        stage('示例') {
            steps {
                script {
                    myCustomFunction()
                }
            }
        }
    }
}
```

# Jenkinsfile 语法
## 框架
```