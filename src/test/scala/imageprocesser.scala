/**
  * Created by jason on 3/28/17.
  */
import org.imgscalr._
import java.io.File
import javax.imageio.ImageIO;
object imageprocesser {

  def makeSquare(img: java.awt.image.BufferedImage):
    java.awt.image.BufferedImage = {
      val w = img.getWidth
      val h = img.getHeight
      val dim = List(w, h). min
        img match{
          case x if w == h => img
          case x if w > h => Scalr.crop(img, (w-h)/2, 0, dim, dim)
          case x if w < h => Scalr.crop(img, 0, (h-w)/2, dim, dim)
        }
  }
  def resize(img: java.awt.image.BufferedImage, width: Int, height: Int) = {
    Scalr.resize(img, Scalr.Method.BALANCED, width, height)
  }



}
